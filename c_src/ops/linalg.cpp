// Linear algebra: matmul, tensordot, outer, inner, decompositions
// (lu, svd, qr, cholesky, eigh), solvers (solve, solve_triangular),
// and quantization primitives (quantize / dequantize /
// quantized_matmul). The `mode` string selects between MLX's
// `"affine"` (int4/int8) scheme and microscaled variants
// (`"mxfp4"`, `"mxfp8"`, `"nvfp4"` — see MLX's `QuantizationMode` enum).

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/einsum.h>
#include <mlx/mlx.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::to_int_vec;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term matmul_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b)](mx::Stream &s) {
        return wrap(mx::matmul(a->array, b->array, s));
      });
}
FINE_NIF(matmul_nif, 0);

fine::Term tensordot_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    std::vector<int64_t> axes_a,
    std::vector<int64_t> axes_b) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b),
       axes_a = std::move(axes_a),
       axes_b = std::move(axes_b)](mx::Stream &s) {
        return wrap(mx::tensordot(a->array, b->array, to_int_vec(axes_a),
                                  to_int_vec(axes_b), s));
      });
}
FINE_NIF(tensordot_nif, 0);

fine::Term outer_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b)](mx::Stream &s) {
        return wrap(mx::outer(a->array, b->array, s));
      });
}
FINE_NIF(outer_nif, 0);

fine::Term inner_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b)](mx::Stream &s) {
        return wrap(mx::inner(a->array, b->array, s));
      });
}
FINE_NIF(inner_nif, 0);

// `biases` is returned by MLX's `quantize` only for the affine mode
// (see `affine_quantize` vs `fp_quantize` in `vendor/mlx/mlx/ops.cpp`).
// For microscaled modes we return a scalar-zero placeholder in the
// biases slot so the Elixir-side QuantizedWeight container keeps a
// uniform three-field tensor shape (Nx.Container can't hold `nil`).
// The dequantize / matmul NIFs take biases as an `std::optional` and
// the Elixir layer passes `nil` for microscaled modes, so the
// placeholder never crosses back into MLX.
static fine::ResourcePtr<Tensor> wrap_biases_or_placeholder(
    std::vector<mx::array>& v) {
  if (v.size() > 2) {
    return wrap(std::move(v[2]));
  }
  return wrap(mx::array(0.0f));
}

static std::optional<mx::array> opt_array(
    const std::optional<fine::ResourcePtr<Tensor>>& opt) {
  if (opt) return (*opt)->array;
  return std::nullopt;
}

// Variadic-operand einsum. MLX picks its own contraction path internally;
// we just pass through the equation string and the operand arrays.
fine::Term einsum_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::string subscripts,
    std::vector<fine::ResourcePtr<Tensor>> operands) {
  return async_encoded(env, w,
      [subscripts = std::move(subscripts),
       operands = std::move(operands)](mx::Stream &s) {
        std::vector<mx::array> operand_arrays;
        operand_arrays.reserve(operands.size());
        for (auto &op : operands) {
          operand_arrays.push_back(op->array);
        }
        return wrap(mx::einsum(subscripts, operand_arrays, s));
      });
}
FINE_NIF(einsum_nif, 0);

fine::Term quantize_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> wt,
    int64_t group_size,
    int64_t bits,
    std::string mode) {
  return async_encoded(env, w,
      [wt = std::move(wt), group_size, bits,
       mode = std::move(mode)](mx::Stream &s) {
        auto result = mx::quantize(wt->array, static_cast<int>(group_size),
                                   static_cast<int>(bits), mode,
                                   std::nullopt, s);
        auto q = wrap(std::move(result[0]));
        auto scales = wrap(std::move(result[1]));
        auto biases = wrap_biases_or_placeholder(result);
        return std::make_tuple(std::move(q), std::move(scales),
                               std::move(biases));
      });
}
FINE_NIF(quantize_nif, 0);

fine::Term dequantize_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> w_q,
    fine::ResourcePtr<Tensor> scales,
    std::optional<fine::ResourcePtr<Tensor>> biases,
    int64_t group_size,
    int64_t bits,
    std::string mode) {
  return async_encoded(env, w,
      [w_q = std::move(w_q), scales = std::move(scales),
       biases = std::move(biases), group_size, bits,
       mode = std::move(mode)](mx::Stream &s) {
        return wrap(mx::dequantize(w_q->array, scales->array, opt_array(biases),
                                   static_cast<int>(group_size),
                                   static_cast<int>(bits), mode,
                                   std::nullopt, std::nullopt, s));
      });
}
FINE_NIF(dequantize_nif, 0);

fine::Term quantized_matmul_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    fine::ResourcePtr<Tensor> w_q,
    fine::ResourcePtr<Tensor> scales,
    std::optional<fine::ResourcePtr<Tensor>> biases,
    bool transpose,
    int64_t group_size,
    int64_t bits,
    std::string mode) {
  return async_encoded(env, w,
      [x = std::move(x), w_q = std::move(w_q), scales = std::move(scales),
       biases = std::move(biases), transpose, group_size, bits,
       mode = std::move(mode)](mx::Stream &s) {
        return wrap(mx::quantized_matmul(x->array, w_q->array, scales->array,
                                         opt_array(biases), transpose,
                                         static_cast<int>(group_size),
                                         static_cast<int>(bits), mode, s));
      });
}
FINE_NIF(quantized_matmul_nif, 0);

// ---- Decompositions / solvers (mx::linalg::*) ------------------
//
// MLX's linalg primitives are CPU-only — they throw on a GPU stream.
// Each NIF dispatches via the worker but forces a CPU stream for the
// actual linalg call. MLX handles cross-stream data dependencies
// internally via its lazy eval graph.

fine::Term linalg_lu_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a) {
  return async_encoded(env, w, [a = std::move(a)](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto result = mx::linalg::lu(a->array, cpu);
    return std::make_tuple(
        wrap(std::move(result[0])),
        wrap(std::move(result[1])),
        wrap(std::move(result[2])));
  });
}
FINE_NIF(linalg_lu_nif, 0);

fine::Term linalg_svd_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a) {
  return async_encoded(env, w, [a = std::move(a)](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto result = mx::linalg::svd(a->array, true, cpu);
    return std::make_tuple(
        wrap(std::move(result[0])),
        wrap(std::move(result[1])),
        wrap(std::move(result[2])));
  });
}
FINE_NIF(linalg_svd_nif, 0);

fine::Term linalg_qr_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a) {
  return async_encoded(env, w, [a = std::move(a)](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto [q, r] = mx::linalg::qr(a->array, cpu);
    return std::make_tuple(wrap(std::move(q)), wrap(std::move(r)));
  });
}
FINE_NIF(linalg_qr_nif, 0);

fine::Term linalg_cholesky_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    bool upper) {
  return async_encoded(env, w,
      [a = std::move(a), upper](mx::Stream & /*s*/) {
        auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
        return wrap(mx::linalg::cholesky(a->array, upper, cpu));
      });
}
FINE_NIF(linalg_cholesky_nif, 0);

fine::Term linalg_eigh_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::string uplo) {
  return async_encoded(env, w,
      [a = std::move(a), uplo = std::move(uplo)](mx::Stream & /*s*/) {
        auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
        auto [vals, vecs] = mx::linalg::eigh(a->array, uplo, cpu);
        return std::make_tuple(wrap(std::move(vals)), wrap(std::move(vecs)));
      });
}
FINE_NIF(linalg_eigh_nif, 0);

fine::Term linalg_solve_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b)](mx::Stream & /*s*/) {
        auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
        return wrap(mx::linalg::solve(a->array, b->array, cpu));
      });
}
FINE_NIF(linalg_solve_nif, 0);

fine::Term linalg_solve_triangular_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    bool upper) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b), upper](mx::Stream & /*s*/) {
        auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
        return wrap(mx::linalg::solve_triangular(a->array, b->array, upper, cpu));
      });
}
FINE_NIF(linalg_solve_triangular_nif, 0);

} // namespace
