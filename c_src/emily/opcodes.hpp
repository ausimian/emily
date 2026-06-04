// Opcode registry for the Expr-compiler program IR. Each opcode maps a
// flat IR instruction to a pure op core (emily/op_cores.hpp). The
// `dispatch_op` switch is the replay engine's inner loop.
//
// Wire values are the integers the Elixir lowerer emits — keep this enum
// in lockstep with `Emily.IR`'s opcode table in lib/emily/ir.ex.
//
// CM0 ships only `Add`; CM1 fills in the full primitive set.

#pragma once

#include "op_cores.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace emily {

namespace mx = mlx::core;

enum class Opcode : int64_t {
  Add = 0,
};

inline constexpr int64_t kOpcodeCount = 1;

inline bool valid_opcode(int64_t v) { return v >= 0 && v < kOpcodeCount; }

// Replay one instruction: apply `op` to its already-resolved operands.
// Operand arity is checked here so a malformed IR raises a clear error
// instead of tripping an MLX assertion deeper in.
inline mx::array dispatch_op(Opcode op, const std::vector<mx::array> &in,
                             mx::Stream &s) {
  switch (op) {
  case Opcode::Add:
    if (in.size() != 2) {
      throw std::invalid_argument("add expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    return emily::ops::add_core(in[0], in[1], s);
  }
  throw std::invalid_argument("unknown opcode " +
                              std::to_string(static_cast<int64_t>(op)));
}

} // namespace emily
