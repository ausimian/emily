// Linear algebra: matmul, tensordot, outer, inner, and affine int4/int8
// quantization primitives (quantize / dequantize / quantized_matmul).

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_int_vec;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> matmul(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return wrap(mx::matmul(a->array, b->array));
}
FINE_NIF(matmul, 0);

// tensordot/4: contract `a` over `axes_a` against `b` over `axes_b`.
fine::ResourcePtr<Tensor> tensordot(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    std::vector<int64_t> axes_a,
    std::vector<int64_t> axes_b) {
  return wrap(mx::tensordot(
      a->array, b->array, to_int_vec(axes_a), to_int_vec(axes_b)));
}
FINE_NIF(tensordot, 0);

fine::ResourcePtr<Tensor> outer(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return wrap(mx::outer(a->array, b->array));
}
FINE_NIF(outer, 0);

fine::ResourcePtr<Tensor> inner(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return wrap(mx::inner(a->array, b->array));
}
FINE_NIF(inner, 0);

// Affine quantization along the last axis.
//
// Returns {w_q, scales, biases}:
//   - w_q is packed uint32 with (last_dim * bits / 32) elements per row
//   - scales and biases share shape (..., last_dim / group_size), dtype
//     matching the input.
//
// MLX requires last_dim % group_size == 0; bits ∈ {2, 3, 4, 6, 8}.
std::tuple<fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>>
quantize(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> w,
    int64_t group_size,
    int64_t bits) {
  auto triple = mx::quantize(
      w->array, static_cast<int>(group_size), static_cast<int>(bits));
  return std::make_tuple(
      wrap(std::move(std::get<0>(triple))),
      wrap(std::move(std::get<1>(triple))),
      wrap(std::move(std::get<2>(triple))));
}
FINE_NIF(quantize, 0);

// Inverse of quantize. Reconstructs a dense tensor from packed w_q plus
// per-group scales and biases.
fine::ResourcePtr<Tensor> dequantize(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> w_q,
    fine::ResourcePtr<Tensor> scales,
    fine::ResourcePtr<Tensor> biases,
    int64_t group_size,
    int64_t bits) {
  return wrap(mx::dequantize(
      w_q->array,
      scales->array,
      biases->array,
      static_cast<int>(group_size),
      static_cast<int>(bits)));
}
FINE_NIF(dequantize, 0);

// Matmul against a quantized weight. `transpose` is wired through
// explicitly because AWQ-style packed checkpoints ship in a different
// layout than freshly-quantized weights, and MLX's kernel selection
// depends on the flag.
fine::ResourcePtr<Tensor> quantized_matmul(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> x,
    fine::ResourcePtr<Tensor> w_q,
    fine::ResourcePtr<Tensor> scales,
    fine::ResourcePtr<Tensor> biases,
    bool transpose,
    int64_t group_size,
    int64_t bits) {
  return wrap(mx::quantized_matmul(
      x->array,
      w_q->array,
      scales->array,
      biases->array,
      transpose,
      static_cast<int>(group_size),
      static_cast<int>(bits)));
}
FINE_NIF(quantized_matmul, 0);

} // namespace
