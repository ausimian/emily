// Program: a compiled, replayable Nx.Defn.Expr graph for the Expr->MLX
// single-NIF compiler.
//
// Built once by `compile_program` (parses the flat IR and captures
// strong refs to weight/const tensors); replayed per call by
// `eval_program` on the worker thread, which rebuilds the `mx::array`
// DAG from the cached instruction list + fresh inputs and evals it.
//
// The win: one BEAM<->worker NIF round-trip per *invocation* instead of
// one per op. Weights cross the boundary once (here) and are never
// re-serialized per eval. See c_src/program.cpp and lib/emily/program.ex.

#pragma once

#include "opcodes.hpp"
#include "tensor.hpp"

#include <fine.hpp>

#include <cstdint>
#include <vector>

namespace emily {

// Operand references are packed into an int64 by the Elixir lowerer
// (`Emily.IR.pack_ref/1`): the high bits carry the slot kind, the low
// bits the index. Unpacked here during compile-time validation and
// during replay. Keep in sync with lib/emily/ir.ex.
namespace ref {

inline constexpr int kTagShift = 48;
inline constexpr int64_t kIndexMask = (int64_t(1) << kTagShift) - 1;

enum class Kind : int64_t { Input = 0, Capture = 1, Const = 2, Instr = 3 };

inline Kind kind_of(int64_t r) {
  return static_cast<Kind>((r >> kTagShift) & 0x3);
}

inline int64_t index_of(int64_t r) { return r & kIndexMask; }

} // namespace ref

struct CompiledInstr {
  Opcode opcode;
  std::vector<int64_t> operands;             // packed refs
  std::vector<std::vector<int64_t>> iattrs;  // integer attrs (shapes/axes/dtype codes)
};

// One resource per compiled program. `captures` / `consts` hold strong
// BEAM refs so the weight buffers stay alive for the program's lifetime
// (fine's ResourcePtr bumps the refcount exactly like every op capture).
class Program {
public:
  int64_t n_inputs = 0;
  std::vector<fine::ResourcePtr<Tensor>> captures;
  std::vector<fine::ResourcePtr<Tensor>> consts;
  std::vector<CompiledInstr> instrs;
  std::vector<int64_t> outputs; // packed refs
};

} // namespace emily
