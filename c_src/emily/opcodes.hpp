// Opcode registry for the Expr-compiler program IR. Each opcode maps a
// flat IR instruction to its MLX expression. `dispatch_op` is the replay
// engine's inner loop: it receives the already-resolved operand arrays
// plus the instruction's integer attributes (`iattrs` — shapes, axes,
// dtype codes) and returns the result array.
//
// Wire values are the integers the Elixir lowerer emits — keep this enum
// in lockstep with `Emily.IR`'s opcode table in lib/emily/ir.ex. The
// replay calls the same `mlx::core::*` entry points (via the same
// emily/tensor.hpp helpers) as the eager per-op NIFs, so the compiled
// path cannot numerically drift from eager.

#pragma once

#include "dtype.hpp"
#include "op_cores.hpp"
#include "tensor.hpp"

#include <mlx/mlx.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace emily {

namespace mx = mlx::core;

enum class Opcode : int64_t {
  // Binary elementwise (arithmetic / bitwise)
  Add = 0,
  Subtract = 1,
  Multiply = 2,
  Divide = 3,
  Power = 4,
  Maximum = 5,
  Minimum = 6,
  Remainder = 7,
  BitwiseAnd = 8,
  BitwiseOr = 9,
  BitwiseXor = 10,
  LeftShift = 11,
  RightShift = 12,
  // Binary compare / logical (produce bool)
  Equal = 13,
  NotEqual = 14,
  Less = 15,
  LessEqual = 16,
  Greater = 17,
  GreaterEqual = 18,
  LogicalAnd = 19,
  LogicalOr = 20,
  // Unary elementwise
  Negative = 21,
  Abs = 22,
  Sign = 23,
  Sqrt = 24,
  Rsqrt = 25,
  Square = 26,
  Reciprocal = 27,
  Exp = 28,
  Log = 29,
  Log1p = 30,
  Sin = 31,
  Cos = 32,
  Tanh = 33,
  Sigmoid = 34,
  Floor = 35,
  Ceil = 36,
  Erf = 37,
  LogicalNot = 38,
  // Cast / shape (carry integer attributes)
  Astype = 39,  // iattrs: [[dtype_code]]
  Reshape = 40, // iattrs: [[d0, d1, ...]]
  Transpose = 41, // iattrs: [[axis0, ...]]
  Squeeze = 42, // iattrs: [[axis0, ...]]
  BroadcastTo = 43, // iattrs: [[d0, ...]]
  // Linear algebra
  Matmul = 44,    // operands: [a, b]
  Tensordot = 45, // operands: [a, b]; iattrs: [[axes_a], [axes_b]]
  // Reductions (iattrs: [[axes...], [keepdims]])
  Sum = 46,
  Prod = 47,
  ReduceMax = 48,
  ReduceMin = 49,
  All = 50,
  Any = 51,
};

inline constexpr int64_t kOpcodeCount = 52;

inline bool valid_opcode(int64_t v) { return v >= 0 && v < kOpcodeCount; }

namespace __op {

inline const mx::array &arg1(const std::vector<mx::array> &in,
                             const char *name) {
  if (in.size() != 1) {
    throw std::invalid_argument(std::string(name) + " expects 1 operand, got " +
                                std::to_string(in.size()));
  }
  return in[0];
}

inline void need2(const std::vector<mx::array> &in, const char *name) {
  if (in.size() != 2) {
    throw std::invalid_argument(std::string(name) + " expects 2 operands, got " +
                                std::to_string(in.size()));
  }
}

inline const std::vector<int64_t> &attr0(const std::vector<std::vector<int64_t>> &a,
                                         const char *name) {
  if (a.empty()) {
    throw std::invalid_argument(std::string(name) + " is missing its attributes");
  }
  return a[0];
}

inline int64_t scalar_attr(const std::vector<std::vector<int64_t>> &a,
                           const char *name) {
  const auto &v = attr0(a, name);
  if (v.size() != 1) {
    throw std::invalid_argument(std::string(name) + " expects one attribute value");
  }
  return v[0];
}

inline const std::vector<int64_t> &
attr_at(const std::vector<std::vector<int64_t>> &a, std::size_t i,
        const char *name) {
  if (a.size() <= i) {
    throw std::invalid_argument(std::string(name) + " is missing attribute " +
                                std::to_string(i));
  }
  return a[i];
}

// Reduction keepdims flag lives in iattrs[1] as a single 0/1.
inline bool keepdims_attr(const std::vector<std::vector<int64_t>> &a,
                          const char *name) {
  const auto &v = attr_at(a, 1, name);
  if (v.size() != 1) {
    throw std::invalid_argument(std::string(name) + " keepdims must be one value");
  }
  return v[0] != 0;
}

} // namespace __op

// Replay one instruction: apply `op` to its resolved operands + attrs.
inline mx::array dispatch_op(Opcode op, const std::vector<mx::array> &in,
                             const std::vector<std::vector<int64_t>> &iattrs,
                             mx::Stream &s) {
  using namespace emily::__op;
  switch (op) {
  // --- Binary arithmetic / bitwise ---
  case Opcode::Add:
    need2(in, "add");
    return emily::ops::add_core(in[0], in[1], s);
  case Opcode::Subtract:
    need2(in, "subtract");
    return mx::subtract(in[0], in[1], s);
  case Opcode::Multiply:
    need2(in, "multiply");
    return mx::multiply(in[0], in[1], s);
  case Opcode::Divide:
    need2(in, "divide");
    return mx::divide(in[0], in[1], s);
  case Opcode::Power:
    need2(in, "power");
    return mx::power(in[0], in[1], s);
  case Opcode::Maximum:
    need2(in, "maximum");
    return mx::maximum(in[0], in[1], s);
  case Opcode::Minimum:
    need2(in, "minimum");
    return mx::minimum(in[0], in[1], s);
  case Opcode::Remainder:
    need2(in, "remainder");
    return mx::remainder(in[0], in[1], s);
  case Opcode::BitwiseAnd:
    need2(in, "bitwise_and");
    return mx::bitwise_and(in[0], in[1], s);
  case Opcode::BitwiseOr:
    need2(in, "bitwise_or");
    return mx::bitwise_or(in[0], in[1], s);
  case Opcode::BitwiseXor:
    need2(in, "bitwise_xor");
    return mx::bitwise_xor(in[0], in[1], s);
  case Opcode::LeftShift:
    need2(in, "left_shift");
    return mx::left_shift(in[0], in[1], s);
  case Opcode::RightShift:
    need2(in, "right_shift");
    return mx::right_shift(in[0], in[1], s);
  // --- Binary compare / logical ---
  case Opcode::Equal:
    need2(in, "equal");
    return mx::equal(in[0], in[1], s);
  case Opcode::NotEqual:
    need2(in, "not_equal");
    return mx::not_equal(in[0], in[1], s);
  case Opcode::Less:
    need2(in, "less");
    return mx::less(in[0], in[1], s);
  case Opcode::LessEqual:
    need2(in, "less_equal");
    return mx::less_equal(in[0], in[1], s);
  case Opcode::Greater:
    need2(in, "greater");
    return mx::greater(in[0], in[1], s);
  case Opcode::GreaterEqual:
    need2(in, "greater_equal");
    return mx::greater_equal(in[0], in[1], s);
  case Opcode::LogicalAnd:
    need2(in, "logical_and");
    return mx::logical_and(in[0], in[1], s);
  case Opcode::LogicalOr:
    need2(in, "logical_or");
    return mx::logical_or(in[0], in[1], s);
  // --- Unary ---
  case Opcode::Negative:
    return mx::negative(arg1(in, "negative"), s);
  case Opcode::Abs:
    return mx::abs(arg1(in, "abs"), s);
  case Opcode::Sign:
    return mx::sign(arg1(in, "sign"), s);
  case Opcode::Sqrt:
    return mx::sqrt(arg1(in, "sqrt"), s);
  case Opcode::Rsqrt:
    return mx::rsqrt(arg1(in, "rsqrt"), s);
  case Opcode::Square:
    return mx::square(arg1(in, "square"), s);
  case Opcode::Reciprocal:
    return mx::reciprocal(arg1(in, "reciprocal"), s);
  case Opcode::Exp:
    return mx::exp(arg1(in, "exp"), s);
  case Opcode::Log:
    return mx::log(arg1(in, "log"), s);
  case Opcode::Log1p:
    return mx::log1p(arg1(in, "log1p"), s);
  case Opcode::Sin:
    return mx::sin(arg1(in, "sin"), s);
  case Opcode::Cos:
    return mx::cos(arg1(in, "cos"), s);
  case Opcode::Tanh:
    return mx::tanh(arg1(in, "tanh"), s);
  case Opcode::Sigmoid:
    return mx::sigmoid(arg1(in, "sigmoid"), s);
  case Opcode::Floor:
    return mx::floor(arg1(in, "floor"), s);
  case Opcode::Ceil:
    return mx::ceil(arg1(in, "ceil"), s);
  case Opcode::Erf:
    return mx::erf(arg1(in, "erf"), s);
  case Opcode::LogicalNot:
    return mx::logical_not(arg1(in, "logical_not"), s);
  // --- Cast / shape ---
  case Opcode::Astype:
    return mx::astype(arg1(in, "astype"),
                      emily::to_mlx_dtype_code(scalar_attr(iattrs, "astype")), s);
  case Opcode::Reshape:
    return mx::reshape(arg1(in, "reshape"),
                       emily::to_mlx_shape(attr0(iattrs, "reshape")), s);
  case Opcode::Transpose:
    return mx::transpose(arg1(in, "transpose"),
                         emily::to_int_vec(attr0(iattrs, "transpose")), s);
  case Opcode::Squeeze:
    return mx::squeeze(arg1(in, "squeeze"),
                       emily::to_int_vec(attr0(iattrs, "squeeze")), s);
  case Opcode::BroadcastTo:
    return mx::broadcast_to(arg1(in, "broadcast_to"),
                            emily::to_mlx_shape(attr0(iattrs, "broadcast_to")), s);
  // --- Linear algebra ---
  case Opcode::Matmul:
    need2(in, "matmul");
    return mx::matmul(in[0], in[1], s);
  case Opcode::Tensordot:
    need2(in, "tensordot");
    return mx::tensordot(in[0], in[1],
                         emily::to_int_vec(attr_at(iattrs, 0, "tensordot")),
                         emily::to_int_vec(attr_at(iattrs, 1, "tensordot")), s);
  // --- Reductions ---
  case Opcode::Sum:
    return mx::sum(arg1(in, "sum"), emily::to_int_vec(attr0(iattrs, "sum")),
                   keepdims_attr(iattrs, "sum"), s);
  case Opcode::Prod:
    return mx::prod(arg1(in, "prod"), emily::to_int_vec(attr0(iattrs, "prod")),
                    keepdims_attr(iattrs, "prod"), s);
  case Opcode::ReduceMax:
    return mx::max(arg1(in, "max"), emily::to_int_vec(attr0(iattrs, "max")),
                   keepdims_attr(iattrs, "max"), s);
  case Opcode::ReduceMin:
    return mx::min(arg1(in, "min"), emily::to_int_vec(attr0(iattrs, "min")),
                   keepdims_attr(iattrs, "min"), s);
  case Opcode::All:
    return mx::all(arg1(in, "all"), emily::to_int_vec(attr0(iattrs, "all")),
                   keepdims_attr(iattrs, "all"), s);
  case Opcode::Any:
    return mx::any(arg1(in, "any"), emily::to_int_vec(attr0(iattrs, "any")),
                   keepdims_attr(iattrs, "any"), s);
  }
  throw std::invalid_argument("unknown opcode " +
                              std::to_string(static_cast<int64_t>(op)));
}

} // namespace emily
