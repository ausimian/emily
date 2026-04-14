// Miscellaneous ops that didn't fit elsewhere: clip, roll, softmax,
// logcumsumexp, array_equal.

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

// clip/3 — clip each element to [min, max]. Both bounds are required
// tensors; the caller can broadcast a scalar if only one side is
// interesting.
fine::ResourcePtr<Tensor> clip(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> a_min,
    fine::ResourcePtr<Tensor> a_max) {
  return wrap(mx::clip(a->array, a_min->array, a_max->array));
}
FINE_NIF(clip, 0);

// roll/3 — shift elements `shift` steps along `axis` with wrap-around.
fine::ResourcePtr<Tensor> roll(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t shift,
    int64_t axis) {
  return wrap(mx::roll(
      a->array, static_cast<int>(shift), static_cast<int>(axis)));
}
FINE_NIF(roll, 0);

// softmax along the given axes.
fine::ResourcePtr<Tensor> softmax(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    bool precise) {
  return wrap(mx::softmax(a->array, to_int_vec(axes), precise));
}
FINE_NIF(softmax, 0);

fine::ResourcePtr<Tensor> logcumsumexp(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool reverse,
    bool inclusive) {
  return wrap(mx::logcumsumexp(
      a->array, static_cast<int>(axis), reverse, inclusive));
}
FINE_NIF(logcumsumexp, 0);

// array_equal/2 — returns a scalar bool tensor. Treats NaNs as unequal.
fine::ResourcePtr<Tensor> array_equal(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    bool equal_nan) {
  return wrap(mx::array_equal(a->array, b->array, equal_nan));
}
FINE_NIF(array_equal, 0);

} // namespace
