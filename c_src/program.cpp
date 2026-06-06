// Expr-compiler program NIFs: compile_program / eval_program /
// describe_program.
//
// `compile_program` parses the flat IR (opcodes + packed operand refs)
// into a replayable Program resource, capturing strong refs to the
// weight/const tensors. `eval_program` replays it on the worker thread
// with fresh dynamic inputs and evals (or async-evals) the outputs in a
// single round-trip. `describe_program` reflects the stored IR back for
// round-trip tests.
//
// See c_src/emily/program.hpp for the resource + ref encoding and
// lib/emily/program.ex for the Elixir wrappers.

#include "emily/async.hpp"
#include "emily/opcodes.hpp"
#include "emily/program.hpp"
#include "emily/tensor.hpp"
#include "emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::CompiledInstr;
using emily::Opcode;
using emily::Program;
using emily::Tensor;
using emily::WorkerThread;
using emily::wrap;

FINE_RESOURCE(Program);

namespace {

// Validate a packed operand ref against its slot bounds. For Instr refs,
// require idx < producing-instruction index so the program is a DAG in
// topological order (no forward/cyclic refs) — replay can then trust
// every ref without per-eval bounds checks. `instr_index` is the index
// of the instruction owning this operand, or the instruction count for
// output refs (any prior instruction is a valid output root).
void validate_ref(int64_t r, int64_t n_inputs, std::size_t n_captures,
                  std::size_t n_consts, int64_t instr_index,
                  const char *where) {
  int64_t idx = emily::ref::index_of(r);
  if (idx < 0) {
    throw std::invalid_argument(std::string(where) + ": negative ref index");
  }

  switch (emily::ref::kind_of(r)) {
  case emily::ref::Kind::Input:
    if (idx >= n_inputs) {
      throw std::invalid_argument(std::string(where) + ": input ref " +
                                  std::to_string(idx) + " out of range (" +
                                  std::to_string(n_inputs) + " inputs)");
    }
    return;
  case emily::ref::Kind::Capture:
    if (static_cast<std::size_t>(idx) >= n_captures) {
      throw std::invalid_argument(std::string(where) + ": capture ref " +
                                  std::to_string(idx) + " out of range");
    }
    return;
  case emily::ref::Kind::Const:
    if (static_cast<std::size_t>(idx) >= n_consts) {
      throw std::invalid_argument(std::string(where) + ": const ref " +
                                  std::to_string(idx) + " out of range");
    }
    return;
  case emily::ref::Kind::Instr:
    if (idx >= instr_index) {
      throw std::invalid_argument(
          std::string(where) + ": instr ref " + std::to_string(idx) +
          " is not a prior instruction (forward or cyclic ref)");
    }
    return;
  }
  throw std::invalid_argument(std::string(where) + ": invalid ref kind");
}

// Replay context: threads the opt-in loop-fusion flag (and the worker handle
// the compiled-fn cache needs for thread-affine teardown) through the
// recursive replay so the `while` arm can fuse its body. Default = no fusion:
// the plain sync/async/build replay, and the inside of any top-level
// mx::compile trace (which is already fused as a whole, and is while-free).
struct ReplayCtx {
  bool fuse_loops = false;
  std::weak_ptr<emily::State> worker;
};

std::vector<mx::array> replay_program(const Program &prog,
                                      const std::vector<mx::array> &inputs,
                                      mx::Stream &s, const ReplayCtx &ctx = {});

// True iff any *top-level* instruction of `prog` is a `while`. mx::compile
// can't trace a data-dependent host loop, so a (sub-)program containing one
// is replayed host-controlled rather than wrapped whole.
bool contains_top_level_while(const Program &prog) {
  for (const auto &instr : prog.instrs) {
    if (instr.opcode == Opcode::While) {
      return true;
    }
  }
  return false;
}

// Fetch (building on first use) `prog`'s per-stream-cached `mx::compile`'d
// replay callable. The compiled graph bakes in `prog`'s captured weights and
// the stream, so the cache is keyed by stream index; `worker` is stored in
// the entry so `~Program` can post the thread-affine cache erase back to the
// worker that built it. `prog` must be a pure DAG (no `while` — its host loop
// can't be traced); callers guard this. Locks the cache mutex exactly once;
// the returned callable is invariant for a given (prog, stream), so a loop
// can hoist this out of its body and invoke the result directly each step.
Program::CompiledFn
compiled_callable(const Program &prog, mx::Stream &s,
                  const std::weak_ptr<emily::State> &worker) {
  std::lock_guard<std::mutex> lock(prog.compile_mtx);
  auto it = prog.compiled.find(s.index);
  if (it == prog.compiled.end()) {
    // Raw pointer (not ResourcePtr) avoids a Program<->fn cycle; the fn is a
    // member of the Program, so `p` outlives it. The traced replay runs with
    // the default ctx (no fusion): a compiled program is while-free, so its
    // replay is a straight DAG.
    const Program *p = &prog;
    mx::Stream captured = s;
    Program::CompiledFn raw =
        [p, captured](const std::vector<mx::array> &in) mutable {
          return replay_program(*p, in, captured);
        };
    Program::CompiledEntry entry;
    entry.worker = worker;
    entry.fn = mx::compile(std::move(raw));
    it = prog.compiled.emplace(s.index, std::move(entry)).first;
  }
  return it->second.fn;
}

// One-shot fused replay: fetch the compiled callable and invoke it. Fuses the
// elementwise runs the raw replay leaves as separate kernels (rms-norm,
// softmax, SiLU gating, residual adds). Returns the still-lazy roots; the
// caller evals them. (Loops should call `compiled_callable` once and invoke
// the result per iteration instead — see the `while` arm.)
std::vector<mx::array>
compiled_replay(const Program &prog, const std::vector<mx::array> &inputs,
                mx::Stream &s, const std::weak_ptr<emily::State> &worker) {
  return compiled_callable(prog, s, worker)(inputs);
}

// Build the mx::array DAG from `prog` + the dynamic `inputs` (indexed by
// input slot) and return the output roots. Shared by the direct replay
// and the mx::compile'd path. Captures/consts are read from `prog`
// (constant across calls); only `inputs` varies. `ctx` carries the opt-in
// while-body fusion (off by default; see ReplayCtx).
std::vector<mx::array> replay_program(const Program &prog,
                                      const std::vector<mx::array> &inputs,
                                      mx::Stream &s, const ReplayCtx &ctx) {
  std::vector<mx::array> values;
  values.reserve(prog.instrs.size());

  auto resolve = [&](int64_t r) -> mx::array {
    int64_t idx = emily::ref::index_of(r);
    switch (emily::ref::kind_of(r)) {
    case emily::ref::Kind::Input:
      return inputs[idx];
    case emily::ref::Kind::Capture:
      return prog.captures[idx]->array;
    case emily::ref::Kind::Const:
      return prog.consts[idx]->array;
    case emily::ref::Kind::Instr:
      return values[idx];
    }
    throw std::runtime_error("eval_program: invalid ref kind");
  };

  for (const auto &instr : prog.instrs) {
    // `while` is the one multi-output, subprogram-carrying op: handle it
    // here rather than in dispatch_op (which returns a single array and
    // sees no subprograms). It reserves `arity` slots in `values` so the
    // `{:instr, base + i}` refs its `:elem` consumers were lowered to
    // resolve correctly.
    if (instr.opcode == Opcode::While) {
      std::vector<mx::array> state;
      state.reserve(instr.operands.size());
      for (auto r : instr.operands) {
        state.push_back(resolve(r));
      }

      const Program &cond_p = *instr.subprograms.at(0);
      const Program &body_p = *instr.subprograms.at(1);

      // Opt-in (CM14): replay the body through a per-stream-cached
      // mx::compile'd callable so the elementwise runs the raw replay leaves
      // as separate kernels fuse — the CM6 win, now under a host-controlled
      // decode loop mx::compile can't trace as a whole. The body is
      // shape-stable (`offset` is a runtime input), so the compiled callable
      // cache-hits across iterations rather than recompiling per step. A body
      // that itself contains a `while` can't be traced, so it stays
      // host-controlled (its own inner body still fuses). The condition is a
      // trivial scalar we must `mx::eval` every step anyway, so it is left as
      // a raw replay regardless of `ctx.fuse_loops`.
      // Build the fused body callable ONCE, before the loop: it is invariant
      // across iterations (same body program + stream), so hoisting it keeps
      // the per-token loop from re-locking the compile cache and copying the
      // std::function every step. Empty when not fusing -> raw replay below.
      const bool fuse_body =
          ctx.fuse_loops && !contains_top_level_while(body_p);
      Program::CompiledFn body_fn =
          fuse_body ? compiled_callable(body_p, s, ctx.worker)
                    : Program::CompiledFn{};

      // Host-controlled loop, entirely on this worker thread — no BEAM
      // round-trip per iteration. Mirrors Nx.Defn.Evaluator: evaluate the
      // condition *before* the body, so zero iterations returns the initial
      // state unchanged. The loop-carried state binds as each subprogram's
      // inputs (`{:input, i}` -> state[i]).
      while (true) {
        std::vector<mx::array> pred = replay_program(cond_p, state, s, ctx);
        mx::array go = mx::astype(pred.at(0), mx::uint8, s);
        mx::eval(go);
        if (go.item<uint8_t>() == 0) {
          break;
        }
        state = body_fn ? body_fn(state)
                        : replay_program(body_p, state, s, ctx);
        // Force the whole next state each iteration so the lazy graph (and
        // memory) stays bounded by the state size, not the trip count.
        mx::eval(state);
      }

      for (auto &v : state) {
        values.push_back(std::move(v));
      }
      continue;
    }

    std::vector<mx::array> operands;
    operands.reserve(instr.operands.size());
    for (auto r : instr.operands) {
      operands.push_back(resolve(r));
    }
    values.push_back(
        emily::dispatch_op(instr.opcode, operands, instr.iattrs, s));
  }

  std::vector<mx::array> roots;
  roots.reserve(prog.outputs.size());
  for (auto r : prog.outputs) {
    roots.push_back(resolve(r));
  }
  return roots;
}

} // namespace

// compile_program/8 — parse the flat IR into a replayable Program
// resource, capturing strong refs to weight/const tensors and to any
// per-instruction child programs (`while` carries [condition, body]).
// Pure bookkeeping: no MLX work, so it runs on a regular scheduler (no
// worker) and returns the resource synchronously.
fine::ResourcePtr<Program>
compile_program(ErlNifEnv *, int64_t n_inputs,
                std::vector<fine::ResourcePtr<Tensor>> captures,
                std::vector<fine::ResourcePtr<Tensor>> consts,
                std::vector<int64_t> opcodes,
                std::vector<std::vector<int64_t>> operands,
                std::vector<std::vector<std::vector<int64_t>>> iattrs,
                std::vector<int64_t> outputs,
                // Per-instruction child programs (already compiled in Elixir,
                // so recursion lives there, not here). Empty for every op but
                // `while`. May be globally empty when no instruction carries
                // any (the common case + the direct-NIF tests).
                std::vector<std::vector<fine::ResourcePtr<Program>>>
                    subprograms) {
  if (n_inputs < 0) {
    throw std::invalid_argument(
        "compile_program: n_inputs must be non-negative, got " +
        std::to_string(n_inputs));
  }
  if (opcodes.size() != operands.size() || opcodes.size() != iattrs.size()) {
    throw std::invalid_argument(
        "compile_program: opcodes/operands/iattrs length mismatch (" +
        std::to_string(opcodes.size()) + " / " +
        std::to_string(operands.size()) + " / " +
        std::to_string(iattrs.size()) + ")");
  }

  auto prog = fine::make_resource<Program>();
  prog->n_inputs = n_inputs;
  prog->captures = std::move(captures);
  prog->consts = std::move(consts);
  prog->instrs.reserve(opcodes.size());

  // `slot` counts value slots produced so far, which differs from the
  // instruction index once a multi-output op is present: `while` produces
  // one value per loop-carried state element (= its operand count), so its
  // outputs occupy `slot .. slot + arity`. An `{:instr, idx}` ref is valid
  // iff it names an already-produced slot (`idx < slot`), so validation
  // tracks `slot`, not the instruction position.
  int64_t slot = 0;
  for (std::size_t i = 0; i < opcodes.size(); i++) {
    if (!emily::valid_opcode(opcodes[i])) {
      throw std::invalid_argument("compile_program: unknown opcode " +
                                  std::to_string(opcodes[i]) +
                                  " at instruction " + std::to_string(i));
    }
    for (auto r : operands[i]) {
      validate_ref(r, n_inputs, prog->captures.size(), prog->consts.size(),
                   slot, "compile_program operand");
    }

    Opcode op = static_cast<Opcode>(opcodes[i]);
    int64_t out_count =
        (op == Opcode::While) ? static_cast<int64_t>(operands[i].size()) : 1;

    std::vector<fine::ResourcePtr<Program>> sub;
    if (i < subprograms.size()) {
      sub = std::move(subprograms[i]);
    }

    // `while` well-formedness, checked here so a malformed loop is rejected
    // at compile rather than crashing (or silently desyncing the `values`
    // vector) deep inside the replay: it carries exactly [condition, body],
    // the condition yields one scalar, and the body returns one value per
    // loop-carried state element (= operand count = out_count). The Elixir
    // lowerer guarantees this; the check guards direct-NIF / future callers.
    if (op == Opcode::While) {
      if (sub.size() != 2) {
        throw std::invalid_argument(
            "compile_program: while at instruction " + std::to_string(i) +
            " expects 2 subprograms (condition, body), got " +
            std::to_string(sub.size()));
      }
      if (sub[0]->outputs.size() != 1) {
        throw std::invalid_argument(
            "compile_program: while condition must produce 1 output, got " +
            std::to_string(sub[0]->outputs.size()));
      }
      if (static_cast<int64_t>(sub[1]->outputs.size()) != out_count) {
        throw std::invalid_argument(
            "compile_program: while body output count (" +
            std::to_string(sub[1]->outputs.size()) +
            ") must equal the loop-carried state size (" +
            std::to_string(out_count) + ")");
      }
    }

    prog->instrs.push_back(CompiledInstr{op, std::move(operands[i]),
                                         std::move(iattrs[i]), std::move(sub)});
    slot += out_count;
  }

  for (auto r : outputs) {
    validate_ref(r, n_inputs, prog->captures.size(), prog->consts.size(), slot,
                 "compile_program output");
  }
  prog->outputs = std::move(outputs);

  return prog;
}
FINE_NIF(compile_program, 0);

// eval_program_nif/4 — replay the program on the worker thread with
// fresh dynamic `inputs` (in slot order), build the mx::array DAG, and
// return the output handles. One NIF round-trip for the whole graph.
// Async (returns a ref; the worker posts the result back) because MLX
// command encoders are thread-local (see emily/worker.hpp).
//
// `eval_mode` controls what happens to the output roots after the DAG
// is built:
//   * 0 (sync)  — mx::eval: block on the GPU before replying.
//   * 1 (async) — mx::async_eval: hand to the command queue and reply
//                 as soon as it's enqueued (overlapped decode loop).
//   * 2 (build) — no eval: return the lazy graph. Isolates the
//                 build/dispatch cost (the lever the compiler pulls)
//                 and lets a caller async_eval several programs at once.
//   * 3 (compiled) — wrap the replay in mx::compile (cached per stream)
//                 then mx::eval. The secondary encode win; opt-in.
//
// The Elixir wrapper `Emily.Native.eval_program/4` awaits the reply via
// `Emily.Native.Async.call/1`.
fine::Term eval_program_nif(ErlNifEnv *env, fine::ResourcePtr<WorkerThread> w,
                            fine::ResourcePtr<Program> prog,
                            std::vector<fine::ResourcePtr<Tensor>> inputs,
                            int64_t eval_mode) {
  // Validate synchronously so a bad call raises before enqueue.
  if (static_cast<int64_t>(inputs.size()) != prog->n_inputs) {
    throw std::invalid_argument(
        "eval_program: expected " + std::to_string(prog->n_inputs) +
        " inputs, got " + std::to_string(inputs.size()));
  }
  if (eval_mode < 0 || eval_mode > 3) {
    throw std::invalid_argument("eval_program: invalid eval_mode " +
                                std::to_string(eval_mode));
  }

  // A weak handle to this worker, captured before `w` is handed to
  // async_encoded, so the compiled-fn cache can be torn down on this same
  // worker thread when the Program is collected (see Program::~Program).
  auto worker_state = w->weak_state();

  return async_encoded(
      env, w,
      [prog = std::move(prog), inputs = std::move(inputs), eval_mode,
       worker_state](mx::Stream &s) {
        std::vector<mx::array> input_arrays = emily::unwrap_all(inputs);
        std::vector<mx::array> roots;

        if (eval_mode == 3) {
          // Compiled path (opt-in; the *secondary* encode win — the main
          // dispatch collapse is the single-NIF replay itself). A program
          // containing a top-level `while` is a data-dependent host loop
          // mx::compile can't trace as a whole, so it is replayed
          // host-controlled with each loop *body* fused under mx::compile
          // (CM14). A while-free program is wrapped whole — the original CM6
          // path, an mx::compile'd replay cached per stream. Either way the
          // captures stay constant and shapes are stable -> cache hits.
          // `mx::eval` only (no async): compiled callables return lazy roots.
          if (contains_top_level_while(*prog)) {
            ReplayCtx ctx;
            ctx.fuse_loops = true;
            ctx.worker = worker_state;
            roots = replay_program(*prog, input_arrays, s, ctx);
          } else {
            roots = compiled_replay(*prog, input_arrays, s, worker_state);
          }
          mx::eval(roots);
        } else {
          roots = replay_program(*prog, input_arrays, s);
          switch (eval_mode) {
          case 0:
            mx::eval(roots);
            break;
          case 1:
            mx::async_eval(roots);
            break;
          default:
            break; // 2 == build only: leave the graph lazy
          }
        }

        std::vector<fine::ResourcePtr<Tensor>> out;
        out.reserve(roots.size());
        for (auto &root : roots) {
          out.push_back(wrap(root));
        }
        return out;
      });
}
FINE_NIF(eval_program_nif, 0);

// describe_program/1 — reflect a compiled Program's stored IR back to
// Elixir as {n_inputs, n_captures, n_consts, opcodes, operands, iattrs,
// outputs} so round-trip tests can assert (lower -> compile -> describe)
// is the identity on the structural part of the IR.
std::tuple<int64_t, int64_t, int64_t, std::vector<int64_t>,
           std::vector<std::vector<int64_t>>,
           std::vector<std::vector<std::vector<int64_t>>>,
           std::vector<int64_t>>
describe_program(ErlNifEnv *, fine::ResourcePtr<Program> prog) {
  std::vector<int64_t> opcodes;
  std::vector<std::vector<int64_t>> operands;
  std::vector<std::vector<std::vector<int64_t>>> iattrs;
  opcodes.reserve(prog->instrs.size());
  operands.reserve(prog->instrs.size());
  iattrs.reserve(prog->instrs.size());
  for (const auto &instr : prog->instrs) {
    opcodes.push_back(static_cast<int64_t>(instr.opcode));
    operands.push_back(instr.operands);
    iattrs.push_back(instr.iattrs);
  }

  return {prog->n_inputs, static_cast<int64_t>(prog->captures.size()),
          static_cast<int64_t>(prog->consts.size()), opcodes, operands, iattrs,
          prog->outputs};
}
FINE_NIF(describe_program, 0);
