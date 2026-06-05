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

// Build the mx::array DAG from `prog` + the dynamic `inputs` (indexed by
// input slot) and return the output roots. Shared by the direct replay
// and the mx::compile'd path. Captures/consts are read from `prog`
// (constant across calls); only `inputs` varies.
std::vector<mx::array> replay_program(const Program &prog,
                                      const std::vector<mx::array> &inputs,
                                      mx::Stream &s) {
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

      // Host-controlled loop, entirely on this worker thread — no BEAM
      // round-trip per iteration. Mirrors Nx.Defn.Evaluator: evaluate the
      // condition *before* the body, so zero iterations returns the initial
      // state unchanged. The loop-carried state binds as each subprogram's
      // inputs (`{:input, i}` -> state[i]).
      while (true) {
        std::vector<mx::array> pred = replay_program(cond_p, state, s);
        mx::array go = mx::astype(pred.at(0), mx::uint8, s);
        mx::eval(go);
        if (go.item<uint8_t>() == 0) {
          break;
        }
        state = replay_program(body_p, state, s);
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

  // A `while` makes the replay a data-dependent, host-controlled loop that
  // mx::compile cannot trace, so the compiled eval mode degrades to a plain
  // sync replay — identical result, just no kernel fusion. The outermost
  // loop is always a top-level instruction, so scanning the top level is
  // enough. (The normal compiler path uses sync anyway; this guards the
  // opt-in compiled mode against a confusing trace-time failure.)
  if (eval_mode == 3) {
    for (const auto &instr : prog->instrs) {
      if (instr.opcode == Opcode::While) {
        eval_mode = 0;
        break;
      }
    }
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
          // CM6: compiled path. Cache an mx::compile'd replay per stream
          // (the compiled graph bakes in the captured weights + stream).
          // Built lazily under the Program's mutex; the captures stay
          // constant, only `input_arrays` varies, and the per-signature
          // Program means shapes are stable -> cache hits. `mx::eval` only
          // (no async): the compiled callable returns realized roots.
          Program::CompiledFn fn;
          {
            std::lock_guard<std::mutex> lock(prog->compile_mtx);
            auto it = prog->compiled.find(s.index);
            if (it == prog->compiled.end()) {
              // Raw pointer (not ResourcePtr) avoids a Program<->fn cycle;
              // the fn is a member of the Program, so `p` outlives it.
              Program *p = prog.get();
              mx::Stream captured = s;
              Program::CompiledFn raw =
                  [p, captured](const std::vector<mx::array> &in) mutable {
                    return replay_program(*p, in, captured);
                  };
              Program::CompiledEntry entry;
              entry.worker = worker_state;
              entry.fn = mx::compile(std::move(raw));
              it = prog->compiled.emplace(s.index, std::move(entry)).first;
            }
            fn = it->second.fn;
          }
          roots = fn(input_arrays);
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
