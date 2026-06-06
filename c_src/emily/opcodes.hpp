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
#include <cstring>
#include <optional>
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
  // Indexing / selection
  Where = 52, // operands: [cond, x, y]
  Slice = 53, // operands: [a]; iattrs: [[start...], [stop...], [strides...]]
  // Fused transformer kernels (mx::fast::*); float attrs are int64 bit
  // patterns (see Emily.IR.float_bits/1 + f64_from_bits below).
  FastRMSNorm = 54,   // operands: [x, weight];        iattrs: [[eps_bits]]
  FastLayerNorm = 55, // operands: [x, weight, bias];  iattrs: [[eps_bits]]
  // operands [x, offset]; iattrs [[dims],[traditional],[base_bits],[scale_bits]]
  FastRoPE = 56,
  // operands [x, offset, freqs]; iattrs [[dims],[traditional],[scale_bits]]
  FastRoPEFreqs = 57,
  // operands [q, k, v]; iattrs [[scale_bits],[causal]]
  FastSDPA = 58,
  // operands [q, k, v, mask]; iattrs [[scale_bits]]
  FastSDPAMask = 59,
  // operands [x, w_q, scales, biases];
  // iattrs [[transpose],[group_size],[bits],[mode_code]]
  QuantizedMatmul = 60,
  // operands [input, indices(s32)]; iattrs [[axis]]
  Take = 61,
  // operands [t0, t1, ...]; iattrs [[axis]]
  Concatenate = 62,
  // operands [src, update, start(s32 [naxes])]; iattrs [[axes...]] —
  // dynamic put_slice (runtime start indices) via mx::slice_update.
  DynSliceUpdate = 63,
  // operands [input(NHWC), kernel(OHWI)]; iattrs [[stride],[pad_lo],
  // [pad_hi],[kernel_dilation],[input_dilation],[groups],[flip]]
  ConvGeneral = 64,
  // Index-of-extremum reductions. operands [a]; iattrs [[axis],[keepdims]]
  Argmax = 65,
  Argmin = 66,
  // Element-wise clamp. operands [a, lo, hi]
  Clip = 67,
  // Sort along an axis (ascending; the lowerer composes a Flip for :desc).
  // operands [a]; iattrs [[axis]]
  Sort = 68,
  Argsort = 69,
  // Reverse along one axis (negative-stride slice). operands [a]; iattrs [[axis]]
  Flip = 70,
  // Data-dependent loop. operands [s0, s1, ...] (initial loop-carried
  // state); iattrs [[arity]]; subprograms [condition, body]. Produces
  // `arity` outputs (the final state). Handled directly in
  // `replay_program` (it needs the subprograms + multi-output), never via
  // `dispatch_op`.
  While = 71,
  // Reinterpret the bytes as another dtype (Nx.bitcast). operands [a];
  // iattrs [[dtype_code]]
  Bitcast = 72,
  // Inverse error function (for Nx.Random.normal). operands [a]
  ErfInv = 73,
  // Slice with runtime (dynamic) start indices, stride 1 (the threefry/RNG
  // path indexes by a loop counter). operands [a, start(s32 [naxes])];
  // iattrs [[axes...], [slice_sizes...]]
  DynSlice = 74,
  // Inclusive cumulative reductions along an axis. operands [a];
  // iattrs [[axis], [reverse]] (Nx cumulation is always inclusive).
  CumSum = 75,
  CumProd = 76,
  CumMax = 77,
  CumMin = 78,
  // Multi-axis gather. operands [input, idx0, idx1, ...] (one s32 index
  // array per gathered axis); iattrs [[axes...], [slice_sizes...]]
  Gather = 79,
  // Stack tensors along a new axis. operands [t0, t1, ...]; iattrs [[axis]]
  Stack = 80,
  // Gather along one axis with a same-rank s32 index tensor.
  // operands [input, indices]; iattrs [[axis]]
  TakeAlongAxis = 81,
  // Window (pooling) reductions: pad -> sliding-window view -> reduce.
  // operands [input, init_scalar]; iattrs
  // [[window...],[strides...],[pad_lo...],[pad_hi...],[dilations...]]
  WindowSum = 82,
  WindowMax = 83,
  WindowMin = 84,
  WindowProduct = 85,
  // Window select-and-scatter (MaxPool/MinPool backward).
  // operands [input, source, init_scalar]; iattrs
  // [[window...],[strides...],[pad_lo...],[pad_hi...]] (no dilations)
  WindowScatterMax = 86,
  WindowScatterMin = 87,
  // FFT family — n-dimensional transforms over the given sizes/axes, with
  // FFTNorm::Backward (unnormalized), matching Nx + the eager fft NIFs.
  // operands [input]; iattrs [[sizes...], [axes...]].
  Fftn = 88,   // complex/real -> complex
  Ifftn = 89,  // complex -> complex (inverse)
  Rfftn = 90,  // real -> complex (half spectrum)
  Irfftn = 91, // complex half-spectrum -> real
  // Scatter (Nx.indexed_put / indexed_add). operands [target, updates,
  // idx0, ...] (one s32 index array per scattered axis); iattrs [[axes...]].
  Scatter = 92,    // overwrite (last write wins on duplicates)
  ScatterAdd = 93, // accumulate
  // Unary elementwise (round 2 — added alongside the @unary_ops expansion
  // for the missing Nx ops; map to the same mx::* primitives the eager
  // unary NIFs use, see c_src/ops/unary.cpp).
  Expm1 = 94,
  Tan = 95,
  Sinh = 96,
  Cosh = 97,
  Arccos = 98,
  Arcsin = 99,
  Arctan = 100,
  Arccosh = 101,
  Arcsinh = 102,
  Arctanh = 103,
  // Round-half-away-from-zero. Backend hard-codes decimals=0
  // (Nx.round/1 takes no decimals arg); the dispatcher does too.
  Round = 104,
  BitwiseInvert = 105,
  Isnan = 106,
  Isinf = 107,
  Conjugate = 108,
  Real = 109,
  Imag = 110,
  // Binary arithmetic peers of the @arith_binary cluster. arctan2 is the
  // direct Backend mapping (atan2: arctan2); floor_divide is the integer
  // engine behind Nx.quotient (the lowerer routes quotient -> floor_divide
  // matching Emily.Backend.quotient/3).
  Arctan2 = 111,
  FloorDivide = 112,
  // Constant-aware pad: pads `input` with the `pad_value` scalar by `low_pad`
  // and `high_pad` on each `axis`. operands [input, pad_value]; iattrs
  // [[axes...], [lows...], [highs...]]. MLX has no interior dilation, so
  // the lowerer rejects interior > 0 (matches Emily.Backend.pad/4).
  Pad = 113,
  // Solve A x = b (or x A = b) where A is triangular. The Backend handles
  // `transform_a`/`left_side` in the lowerer (via transpose ops on a/b);
  // this opcode is the bare kernel call. operands [a, b]; iattrs [[upper]].
  // Routed to MLX's CPU stream (`mx::linalg::solve_triangular` is CPU-only),
  // mirroring c_src/ops/linalg.cpp's eager NIF.
  LinalgSolveTriangular = 114,
  // CPU-only linalg decompositions / solvers (mx::linalg::*). Each
  // routes the call to the CPU stream like the eager NIFs in
  // c_src/ops/linalg.cpp. Single-output (Cholesky, Solve) dispatches
  // through the standard `dispatch_op` path; multi-output (QR/Eigh/LU/SVD)
  // is special-cased in replay_program — the dispatcher returns one
  // array; multi-output instructions push N values into the program
  // values vector, the same shape `while` uses.
  // operands [a]; iattrs [[upper]]
  LinalgCholesky = 115,
  // operands [a, b]; no iattrs
  LinalgSolve = 116,
  // operands [a]; arity 2 (Q, R); no iattrs
  LinalgQR = 117,
  // operands [a]; arity 2 (vals, vecs); uplo hard-coded "L" to match
  // Emily.Backend.native_eigh/3.
  LinalgEigh = 118,
  // operands [a]; arity 3 (perm[s32], L, U); the IR post-processes the
  // perm vector into a permutation matrix via take(eye, perm, 0),
  // mirroring Emily.Backend.native_lu/3.
  LinalgLU = 119,
  // operands [a]; arity 3 (U, S, V); `full_matrices` hard-coded `true` —
  // the IR slices for the thin case, matching Emily.Backend.native_svd/3.
  LinalgSVD = 120,
};

inline constexpr int64_t kOpcodeCount = 121;

// Quant mode code (Emily.IR @quant_modes) -> MLX mode string.
inline std::string qmode_from_code(int64_t code) {
  switch (code) {
  case 0: return "affine";
  case 1: return "mxfp4";
  case 2: return "mxfp8";
  case 3: return "nvfp4";
  default:
    throw std::invalid_argument("unknown quant mode code " +
                                std::to_string(code));
  }
}

inline bool valid_opcode(int64_t v) { return v >= 0 && v < kOpcodeCount; }

// Decode an IEEE-754 double from the int64 bit pattern carried in iattrs
// (the IR's integer attribute channel). Keep in sync with
// Emily.IR.float_bits/1.
inline double f64_from_bits(int64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(d));
  return d;
}

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

inline const std::vector<int64_t> &
attr_at(const std::vector<std::vector<int64_t>> &a, std::size_t i,
        const char *name) {
  if (a.size() <= i) {
    throw std::invalid_argument(std::string(name) + " is missing attribute " +
                                std::to_string(i));
  }
  return a[i];
}

// The single value of the i-th attribute list (for scalar attrs like
// dims / flags / float bit patterns packed one-per-list).
inline int64_t scalar_at(const std::vector<std::vector<int64_t>> &a,
                         std::size_t i, const char *name) {
  const auto &v = attr_at(a, i, name);
  if (v.size() != 1) {
    throw std::invalid_argument(std::string(name) + " attribute " +
                                std::to_string(i) + " must be one value");
  }
  return v[0];
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
                      emily::to_mlx_dtype_code(scalar_at(iattrs, 0, "astype")), s);
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
  // --- Indexing / selection ---
  case Opcode::Where:
    if (in.size() != 3) {
      throw std::invalid_argument("where expects 3 operands, got " +
                                  std::to_string(in.size()));
    }
    return mx::where(in[0], in[1], in[2], s);
  case Opcode::Slice:
    return mx::slice(arg1(in, "slice"),
                     emily::to_mlx_shape(attr_at(iattrs, 0, "slice")),
                     emily::to_mlx_shape(attr_at(iattrs, 1, "slice")),
                     emily::to_mlx_shape(attr_at(iattrs, 2, "slice")), s);
  // --- Fused transformer kernels ---
  case Opcode::FastRMSNorm: {
    if (in.size() != 2) {
      throw std::invalid_argument("fast_rms_norm expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    auto eps = static_cast<float>(
        emily::f64_from_bits(scalar_at(iattrs, 0, "fast_rms_norm")));
    return mx::fast::rms_norm(in[0], std::optional<mx::array>(in[1]), eps, s);
  }
  case Opcode::FastLayerNorm: {
    if (in.size() != 3) {
      throw std::invalid_argument("fast_layer_norm expects 3 operands, got " +
                                  std::to_string(in.size()));
    }
    auto eps = static_cast<float>(
        emily::f64_from_bits(scalar_at(iattrs, 0, "fast_layer_norm")));
    return mx::fast::layer_norm(in[0], std::optional<mx::array>(in[1]),
                                std::optional<mx::array>(in[2]), eps, s);
  }
  case Opcode::FastRoPE: {
    if (in.size() != 2) {
      throw std::invalid_argument("fast_rope expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    int dims = emily::checked_int(scalar_at(iattrs, 0, "fast_rope"), "dims");
    bool traditional = scalar_at(iattrs, 1, "fast_rope") != 0;
    std::optional<float> base =
        static_cast<float>(emily::f64_from_bits(scalar_at(iattrs, 2, "fast_rope")));
    auto scale =
        static_cast<float>(emily::f64_from_bits(scalar_at(iattrs, 3, "fast_rope")));
    return mx::fast::rope(in[0], dims, traditional, base, scale, in[1],
                          std::nullopt, s);
  }
  case Opcode::FastRoPEFreqs: {
    if (in.size() != 3) {
      throw std::invalid_argument("fast_rope_freqs expects 3 operands, got " +
                                  std::to_string(in.size()));
    }
    int dims =
        emily::checked_int(scalar_at(iattrs, 0, "fast_rope_freqs"), "dims");
    bool traditional = scalar_at(iattrs, 1, "fast_rope_freqs") != 0;
    auto scale = static_cast<float>(
        emily::f64_from_bits(scalar_at(iattrs, 2, "fast_rope_freqs")));
    return mx::fast::rope(in[0], dims, traditional, std::nullopt, scale, in[1],
                          std::optional<mx::array>(in[2]), s);
  }
  case Opcode::FastSDPA: {
    if (in.size() != 3) {
      throw std::invalid_argument("fast_sdpa expects 3 operands, got " +
                                  std::to_string(in.size()));
    }
    auto scale =
        static_cast<float>(emily::f64_from_bits(scalar_at(iattrs, 0, "fast_sdpa")));
    std::string mask_mode = scalar_at(iattrs, 1, "fast_sdpa") != 0 ? "causal" : "";
    return mx::fast::scaled_dot_product_attention(
        in[0], in[1], in[2], scale, mask_mode, std::nullopt, std::nullopt, s);
  }
  case Opcode::FastSDPAMask: {
    if (in.size() != 4) {
      throw std::invalid_argument("fast_sdpa_mask expects 4 operands, got " +
                                  std::to_string(in.size()));
    }
    auto scale = static_cast<float>(
        emily::f64_from_bits(scalar_at(iattrs, 0, "fast_sdpa_mask")));
    return mx::fast::scaled_dot_product_attention(
        in[0], in[1], in[2], scale, "array", std::optional<mx::array>(in[3]),
        std::nullopt, s);
  }
  case Opcode::QuantizedMatmul: {
    if (in.size() != 4) {
      throw std::invalid_argument("quantized_matmul expects 4 operands, got " +
                                  std::to_string(in.size()));
    }
    bool transpose = scalar_at(iattrs, 0, "quantized_matmul") != 0;
    int gs = emily::checked_int(scalar_at(iattrs, 1, "quantized_matmul"),
                                "group_size");
    int bits =
        emily::checked_int(scalar_at(iattrs, 2, "quantized_matmul"), "bits");
    std::string mode =
        emily::qmode_from_code(scalar_at(iattrs, 3, "quantized_matmul"));
    // Affine carries real biases; microscaled modes pass nullopt (the
    // 4th operand is a placeholder), matching Native.quantized_matmul.
    std::optional<mx::array> biases;
    if (mode == "affine") {
      biases = in[3];
    }
    return mx::quantized_matmul(in[0], in[1], in[2], biases, transpose, gs, bits,
                                mode, s);
  }
  case Opcode::Take: {
    if (in.size() != 2) {
      throw std::invalid_argument("take expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    int axis = emily::checked_int(scalar_at(iattrs, 0, "take"), "axis");
    return mx::take(in[0], in[1], axis, s);
  }
  case Opcode::TakeAlongAxis: {
    if (in.size() != 2) {
      throw std::invalid_argument("take_along_axis expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    int axis =
        emily::checked_int(scalar_at(iattrs, 0, "take_along_axis"), "axis");
    return mx::take_along_axis(in[0], in[1], axis, s);
  }
  case Opcode::WindowSum:
  case Opcode::WindowMax:
  case Opcode::WindowMin:
  case Opcode::WindowProduct: {
    if (in.size() != 2) {
      throw std::invalid_argument(
          "window reduce expects 2 operands (input, init), got " +
          std::to_string(in.size()));
    }
    auto kind = op == Opcode::WindowSum    ? emily::ops::WindowReduceKind::Sum
                : op == Opcode::WindowMax  ? emily::ops::WindowReduceKind::Max
                : op == Opcode::WindowMin  ? emily::ops::WindowReduceKind::Min
                                           : emily::ops::WindowReduceKind::Product;
    return emily::ops::window_reduce_core(
        in[0], attr_at(iattrs, 0, "window"), attr_at(iattrs, 1, "window"),
        attr_at(iattrs, 2, "window"), attr_at(iattrs, 3, "window"),
        attr_at(iattrs, 4, "window"), in[1], kind, s);
  }
  case Opcode::WindowScatterMax:
  case Opcode::WindowScatterMin: {
    if (in.size() != 3) {
      throw std::invalid_argument(
          "window scatter expects 3 operands (input, source, init), got " +
          std::to_string(in.size()));
    }
    bool is_max = op == Opcode::WindowScatterMax;
    return emily::ops::window_scatter_core(
        in[0], in[1], in[2], attr_at(iattrs, 0, "window_scatter"),
        attr_at(iattrs, 1, "window_scatter"),
        attr_at(iattrs, 2, "window_scatter"),
        attr_at(iattrs, 3, "window_scatter"), is_max, s);
  }
  case Opcode::Concatenate: {
    if (in.empty()) {
      throw std::invalid_argument("concatenate expects >= 1 operand");
    }
    int axis = emily::checked_int(scalar_at(iattrs, 0, "concatenate"), "axis");
    return mx::concatenate(in, axis, s);
  }
  case Opcode::DynSliceUpdate: {
    if (in.size() != 3) {
      throw std::invalid_argument("dyn_slice_update expects 3 operands, got " +
                                  std::to_string(in.size()));
    }
    return mx::slice_update(in[0], in[1], in[2],
                            emily::to_int_vec(attr0(iattrs, "dyn_slice_update")),
                            s);
  }
  case Opcode::ConvGeneral: {
    need2(in, "conv_general");
    int groups =
        emily::checked_int(scalar_at(iattrs, 5, "conv_general"), "groups");
    bool flip = scalar_at(iattrs, 6, "conv_general") != 0;
    return mx::conv_general(
        in[0], in[1], emily::to_int_vec(attr_at(iattrs, 0, "conv_general")),
        emily::to_int_vec(attr_at(iattrs, 1, "conv_general")),
        emily::to_int_vec(attr_at(iattrs, 2, "conv_general")),
        emily::to_int_vec(attr_at(iattrs, 3, "conv_general")),
        emily::to_int_vec(attr_at(iattrs, 4, "conv_general")), groups, flip, s);
  }
  // --- Selection / sort ---
  case Opcode::Argmax:
    return mx::argmax(arg1(in, "argmax"),
                      emily::checked_int(scalar_at(iattrs, 0, "argmax"), "axis"),
                      keepdims_attr(iattrs, "argmax"), s);
  case Opcode::Argmin:
    return mx::argmin(arg1(in, "argmin"),
                      emily::checked_int(scalar_at(iattrs, 0, "argmin"), "axis"),
                      keepdims_attr(iattrs, "argmin"), s);
  case Opcode::Clip:
    if (in.size() != 3) {
      throw std::invalid_argument("clip expects 3 operands, got " +
                                  std::to_string(in.size()));
    }
    return mx::clip(in[0], in[1], in[2], s);
  case Opcode::Sort:
    return mx::sort(arg1(in, "sort"),
                    emily::checked_int(scalar_at(iattrs, 0, "sort"), "axis"), s);
  case Opcode::Argsort:
    return mx::argsort(arg1(in, "argsort"),
                       emily::checked_int(scalar_at(iattrs, 0, "argsort"), "axis"),
                       s);
  case Opcode::Flip:
    // Reverse along one axis — shared core with the eager flip_nif.
    return emily::ops::flip_core(arg1(in, "flip"), scalar_at(iattrs, 0, "flip"),
                                 s);
  case Opcode::While:
    // Multi-output + carries subprograms; handled directly in
    // replay_program, never dispatched here.
    throw std::invalid_argument("while is handled in replay_program");
  case Opcode::Bitcast:
    return mx::view(arg1(in, "bitcast"),
                    emily::to_mlx_dtype_code(scalar_at(iattrs, 0, "bitcast")), s);
  case Opcode::ErfInv:
    return mx::erfinv(arg1(in, "erf_inv"), s);
  case Opcode::DynSlice:
    // operands [a, start(s32 [naxes])]; iattrs [[axes...], [slice_sizes...]].
    // Stride-1 dynamic slice (mx::slice's dynamic-start overload).
    if (in.size() != 2) {
      throw std::invalid_argument("dyn_slice expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    return mx::slice(in[0], in[1],
                     emily::to_int_vec(attr_at(iattrs, 0, "dyn_slice")),
                     emily::to_mlx_shape(attr_at(iattrs, 1, "dyn_slice")), s);
  case Opcode::CumSum:
    return mx::cumsum(arg1(in, "cumsum"),
                      emily::checked_int(scalar_at(iattrs, 0, "cumsum"), "axis"),
                      scalar_at(iattrs, 1, "cumsum") != 0, /*inclusive=*/true, s);
  case Opcode::CumProd:
    return mx::cumprod(arg1(in, "cumprod"),
                       emily::checked_int(scalar_at(iattrs, 0, "cumprod"), "axis"),
                       scalar_at(iattrs, 1, "cumprod") != 0, /*inclusive=*/true, s);
  case Opcode::CumMax:
    return mx::cummax(arg1(in, "cummax"),
                      emily::checked_int(scalar_at(iattrs, 0, "cummax"), "axis"),
                      scalar_at(iattrs, 1, "cummax") != 0, /*inclusive=*/true, s);
  case Opcode::CumMin:
    return mx::cummin(arg1(in, "cummin"),
                      emily::checked_int(scalar_at(iattrs, 0, "cummin"), "axis"),
                      scalar_at(iattrs, 1, "cummin") != 0, /*inclusive=*/true, s);
  case Opcode::Gather: {
    if (in.size() < 2) {
      throw std::invalid_argument("gather expects input + >=1 index operand");
    }
    std::vector<mx::array> indices(in.begin() + 1, in.end());
    return mx::gather(in[0], indices,
                      emily::to_int_vec(attr_at(iattrs, 0, "gather")),
                      emily::to_mlx_shape(attr_at(iattrs, 1, "gather")), s);
  }
  case Opcode::Stack:
    if (in.empty()) {
      throw std::invalid_argument("stack expects >= 1 operand");
    }
    return mx::stack(in, emily::checked_int(scalar_at(iattrs, 0, "stack"), "axis"),
                     s);
  // --- FFT family (shares the eager fft.cpp entry points) ---
  case Opcode::Fftn:
    return mx::fft::fftn(arg1(in, "fftn"),
                         emily::to_mlx_shape(attr_at(iattrs, 0, "fftn")),
                         emily::to_int_vec(attr_at(iattrs, 1, "fftn")),
                         mx::fft::FFTNorm::Backward, s);
  case Opcode::Ifftn:
    return mx::fft::ifftn(arg1(in, "ifftn"),
                          emily::to_mlx_shape(attr_at(iattrs, 0, "ifftn")),
                          emily::to_int_vec(attr_at(iattrs, 1, "ifftn")),
                          mx::fft::FFTNorm::Backward, s);
  case Opcode::Rfftn:
    return mx::fft::rfftn(arg1(in, "rfftn"),
                          emily::to_mlx_shape(attr_at(iattrs, 0, "rfftn")),
                          emily::to_int_vec(attr_at(iattrs, 1, "rfftn")),
                          mx::fft::FFTNorm::Backward, s);
  case Opcode::Irfftn:
    return mx::fft::irfftn(arg1(in, "irfftn"),
                           emily::to_mlx_shape(attr_at(iattrs, 0, "irfftn")),
                           emily::to_int_vec(attr_at(iattrs, 1, "irfftn")),
                           mx::fft::FFTNorm::Backward, s);
  // --- Unary elementwise (round 2) ---
  case Opcode::Expm1:
    return mx::expm1(arg1(in, "expm1"), s);
  case Opcode::Tan:
    return mx::tan(arg1(in, "tan"), s);
  case Opcode::Sinh:
    return mx::sinh(arg1(in, "sinh"), s);
  case Opcode::Cosh:
    return mx::cosh(arg1(in, "cosh"), s);
  case Opcode::Arccos:
    return mx::arccos(arg1(in, "arccos"), s);
  case Opcode::Arcsin:
    return mx::arcsin(arg1(in, "arcsin"), s);
  case Opcode::Arctan:
    return mx::arctan(arg1(in, "arctan"), s);
  case Opcode::Arccosh:
    return mx::arccosh(arg1(in, "arccosh"), s);
  case Opcode::Arcsinh:
    return mx::arcsinh(arg1(in, "arcsinh"), s);
  case Opcode::Arctanh:
    return mx::arctanh(arg1(in, "arctanh"), s);
  case Opcode::Round:
    return mx::round(arg1(in, "round"), /*decimals=*/0, s);
  case Opcode::BitwiseInvert:
    return mx::bitwise_invert(arg1(in, "bitwise_invert"), s);
  case Opcode::Isnan:
    return mx::isnan(arg1(in, "isnan"), s);
  case Opcode::Isinf:
    return mx::isinf(arg1(in, "isinf"), s);
  case Opcode::Conjugate:
    return mx::conjugate(arg1(in, "conjugate"), s);
  case Opcode::Real:
    return mx::real(arg1(in, "real"), s);
  case Opcode::Imag:
    return mx::imag(arg1(in, "imag"), s);
  // --- Binary arithmetic peers ---
  case Opcode::Arctan2:
    need2(in, "arctan2");
    return mx::arctan2(in[0], in[1], s);
  case Opcode::FloorDivide:
    need2(in, "floor_divide");
    return mx::floor_divide(in[0], in[1], s);
  // --- Shape / linalg peers ---
  case Opcode::Pad:
    // operands [input, pad_value]; iattrs [[axes...], [lows...], [highs...]].
    // Mirrors c_src/ops/shape.cpp's pad_nif: `mx::pad` with "constant" mode
    // (interior is rejected up-stack by the Elixir lowerer; same constraint
    // as the eager NIF).
    if (in.size() != 2) {
      throw std::invalid_argument("pad expects 2 operands, got " +
                                  std::to_string(in.size()));
    }
    return mx::pad(in[0], emily::to_int_vec(attr_at(iattrs, 0, "pad")),
                   emily::to_mlx_shape(attr_at(iattrs, 1, "pad")),
                   emily::to_mlx_shape(attr_at(iattrs, 2, "pad")), in[1],
                   "constant", s);
  case Opcode::LinalgSolveTriangular: {
    // mx::linalg::solve_triangular is CPU-only — eager NIF in
    // c_src/ops/linalg.cpp routes to mx::default_stream(cpu) too. The
    // dispatcher's `s` arg is the replay stream (typically GPU); override
    // here so the solver runs on CPU like the eager path.
    need2(in, "linalg_solve_triangular");
    bool upper =
        scalar_at(iattrs, 0, "linalg_solve_triangular") != 0;
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    return mx::linalg::solve_triangular(in[0], in[1], upper, cpu);
  }
  case Opcode::LinalgCholesky: {
    // CPU-only, mirrors c_src/ops/linalg.cpp's eager NIF.
    bool upper = scalar_at(iattrs, 0, "linalg_cholesky") != 0;
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    return mx::linalg::cholesky(arg1(in, "linalg_cholesky"), upper, cpu);
  }
  case Opcode::LinalgSolve: {
    need2(in, "linalg_solve");
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    return mx::linalg::solve(in[0], in[1], cpu);
  }
  // Multi-output linalg (QR/Eigh/LU/SVD) is special-cased in
  // replay_program (it pushes N values per instruction instead of one).
  // Reaching these here would be a compiler bug — surface it loudly
  // rather than silently fall through to "unknown opcode".
  case Opcode::LinalgQR:
  case Opcode::LinalgEigh:
  case Opcode::LinalgLU:
  case Opcode::LinalgSVD:
    throw std::invalid_argument(
        "multi-output linalg op is handled in replay_program");
  // --- Scatter (shares the eager index.cpp entry points) ---
  case Opcode::Scatter:
  case Opcode::ScatterAdd: {
    if (in.size() < 3) {
      throw std::invalid_argument(
          "scatter expects >= 3 operands (target, updates, >=1 index), got " +
          std::to_string(in.size()));
    }
    // operands [target, updates, idx0, ...]; the index arrays follow updates.
    std::vector<mx::array> indices(in.begin() + 2, in.end());
    auto axes = emily::to_int_vec(attr0(iattrs, "scatter"));
    return op == Opcode::Scatter
               ? mx::scatter(in[0], indices, in[1], axes, s)
               : mx::scatter_add(in[0], indices, in[1], axes, s);
  }
  }
  throw std::invalid_argument("unknown opcode " +
                              std::to_string(static_cast<int64_t>(op)));
}

} // namespace emily
