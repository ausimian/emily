// Linear algebra: matmul, tensordot, outer, inner.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
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

} // namespace
