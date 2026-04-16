// Shape manipulation: reshape, transpose, squeeze, expand_dims,
// broadcast_to, concatenate, stack, flatten, pad, tile, swapaxes.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::unwrap_all;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> reshape(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> shape,
    int64_t s) {
  return wrap(mx::reshape(a->array, to_mlx_shape(shape), emily::resolve_stream(s)));
}
FINE_NIF(reshape, 0);

fine::ResourcePtr<Tensor> transpose(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    int64_t s) {
  return wrap(mx::transpose(a->array, to_int_vec(axes), emily::resolve_stream(s)));
}
FINE_NIF(transpose, 0);

fine::ResourcePtr<Tensor> squeeze(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    int64_t s) {
  return wrap(mx::squeeze(a->array, to_int_vec(axes), emily::resolve_stream(s)));
}
FINE_NIF(squeeze, 0);

fine::ResourcePtr<Tensor> expand_dims(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    int64_t s) {
  return wrap(mx::expand_dims(a->array, to_int_vec(axes), emily::resolve_stream(s)));
}
FINE_NIF(expand_dims, 0);

fine::ResourcePtr<Tensor> broadcast_to(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> shape,
    int64_t s) {
  return wrap(mx::broadcast_to(a->array, to_mlx_shape(shape), emily::resolve_stream(s)));
}
FINE_NIF(broadcast_to, 0);

fine::ResourcePtr<Tensor> concatenate(
    ErlNifEnv *,
    std::vector<fine::ResourcePtr<Tensor>> arrays,
    int64_t axis,
    int64_t s) {
  return wrap(mx::concatenate(unwrap_all(arrays), static_cast<int>(axis), emily::resolve_stream(s)));
}
FINE_NIF(concatenate, 0);

fine::ResourcePtr<Tensor> stack(
    ErlNifEnv *,
    std::vector<fine::ResourcePtr<Tensor>> arrays,
    int64_t axis,
    int64_t s) {
  return wrap(mx::stack(unwrap_all(arrays), static_cast<int>(axis), emily::resolve_stream(s)));
}
FINE_NIF(stack, 0);

fine::ResourcePtr<Tensor> flatten(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t start_axis,
    int64_t end_axis,
    int64_t s) {
  return wrap(mx::flatten(
      a->array, static_cast<int>(start_axis), static_cast<int>(end_axis), emily::resolve_stream(s)));
}
FINE_NIF(flatten, 0);

fine::ResourcePtr<Tensor> tile(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> reps,
    int64_t s) {
  return wrap(mx::tile(a->array, to_int_vec(reps), emily::resolve_stream(s)));
}
FINE_NIF(tile, 0);

fine::ResourcePtr<Tensor> swapaxes(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t axis1,
    int64_t axis2,
    int64_t s) {
  return wrap(mx::swapaxes(
      a->array, static_cast<int>(axis1), static_cast<int>(axis2), emily::resolve_stream(s)));
}
FINE_NIF(swapaxes, 0);

// pad/5: per-axis constant pad.
// axes: list of axes to pad.
// low_pad: per-axis low-side padding (same length as axes).
// high_pad: per-axis high-side padding.
// pad_value: scalar tensor with the padding constant.
fine::ResourcePtr<Tensor> pad(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    std::vector<int64_t> low_pad,
    std::vector<int64_t> high_pad,
    fine::ResourcePtr<Tensor> pad_value,
    int64_t s) {
  return wrap(mx::pad(
      a->array,
      to_int_vec(axes),
      to_mlx_shape(low_pad),
      to_mlx_shape(high_pad),
      pad_value->array,
      "constant",
      emily::resolve_stream(s)));
}
FINE_NIF(pad, 0);

// repeat/3: repeat along an axis.
fine::ResourcePtr<Tensor> repeat(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t repeats,
    int64_t axis,
    int64_t s) {
  return wrap(mx::repeat(
      a->array, static_cast<int>(repeats), static_cast<int>(axis), emily::resolve_stream(s)));
}
FINE_NIF(repeat, 0);

} // namespace
