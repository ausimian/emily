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

} // namespace

// compile_program/6 — parse the flat IR into a replayable Program
// resource, capturing strong refs to weight/const tensors. Pure
// bookkeeping: no MLX work, so it runs on a regular scheduler (no
// worker) and returns the resource synchronously.
fine::ResourcePtr<Program>
compile_program(ErlNifEnv *, int64_t n_inputs,
                std::vector<fine::ResourcePtr<Tensor>> captures,
                std::vector<fine::ResourcePtr<Tensor>> consts,
                std::vector<int64_t> opcodes,
                std::vector<std::vector<int64_t>> operands,
                std::vector<std::vector<std::vector<int64_t>>> iattrs,
                std::vector<int64_t> outputs) {
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

  for (std::size_t i = 0; i < opcodes.size(); i++) {
    if (!emily::valid_opcode(opcodes[i])) {
      throw std::invalid_argument("compile_program: unknown opcode " +
                                  std::to_string(opcodes[i]) +
                                  " at instruction " + std::to_string(i));
    }
    for (auto r : operands[i]) {
      validate_ref(r, n_inputs, prog->captures.size(), prog->consts.size(),
                   static_cast<int64_t>(i), "compile_program operand");
    }
    prog->instrs.push_back(CompiledInstr{static_cast<Opcode>(opcodes[i]),
                                         std::move(operands[i]),
                                         std::move(iattrs[i])});
  }

  for (auto r : outputs) {
    validate_ref(r, n_inputs, prog->captures.size(), prog->consts.size(),
                 static_cast<int64_t>(opcodes.size()), "compile_program output");
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
  if (eval_mode < 0 || eval_mode > 2) {
    throw std::invalid_argument("eval_program: invalid eval_mode " +
                                std::to_string(eval_mode));
  }

  return async_encoded(
      env, w,
      [prog = std::move(prog), inputs = std::move(inputs),
       eval_mode](mx::Stream &s) {
        std::vector<mx::array> values;
        values.reserve(prog->instrs.size());

        // Resolve a packed ref to its mx::array. Returns by value — an
        // mx::array copy is a cheap refcount bump, and it avoids any
        // dangling reference into `values` as that vector grows.
        auto resolve = [&](int64_t r) -> mx::array {
          int64_t idx = emily::ref::index_of(r);
          switch (emily::ref::kind_of(r)) {
          case emily::ref::Kind::Input:
            return inputs[idx]->array;
          case emily::ref::Kind::Capture:
            return prog->captures[idx]->array;
          case emily::ref::Kind::Const:
            return prog->consts[idx]->array;
          case emily::ref::Kind::Instr:
            return values[idx];
          }
          throw std::runtime_error("eval_program: invalid ref kind");
        };

        for (const auto &instr : prog->instrs) {
          std::vector<mx::array> operands;
          operands.reserve(instr.operands.size());
          for (auto r : instr.operands) {
            operands.push_back(resolve(r));
          }
          values.push_back(
              emily::dispatch_op(instr.opcode, operands, instr.iattrs, s));
        }

        std::vector<mx::array> roots;
        roots.reserve(prog->outputs.size());
        for (auto r : prog->outputs) {
          roots.push_back(resolve(r));
        }

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
