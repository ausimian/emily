// Shape manipulation: reshape, transpose, squeeze, expand_dims,
// broadcast_to, concatenate, stack, flatten, pad, tile, swapaxes.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::unwrap_all;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term reshape_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> shape) {
  return async_encoded(env, w,
      [a = std::move(a), shape = std::move(shape)](mx::Stream &s) {
        return wrap(mx::reshape(a->array, to_mlx_shape(shape), s));
      });
}
FINE_NIF(reshape_nif, 0);

fine::Term transpose_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), axes = std::move(axes)](mx::Stream &s) {
        return wrap(mx::transpose(a->array, to_int_vec(axes), s));
      });
}
FINE_NIF(transpose_nif, 0);

fine::Term squeeze_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), axes = std::move(axes)](mx::Stream &s) {
        return wrap(mx::squeeze(a->array, to_int_vec(axes), s));
      });
}
FINE_NIF(squeeze_nif, 0);

fine::Term expand_dims_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), axes = std::move(axes)](mx::Stream &s) {
        return wrap(mx::expand_dims(a->array, to_int_vec(axes), s));
      });
}
FINE_NIF(expand_dims_nif, 0);

fine::Term broadcast_to_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> shape) {
  return async_encoded(env, w,
      [a = std::move(a), shape = std::move(shape)](mx::Stream &s) {
        return wrap(mx::broadcast_to(a->array, to_mlx_shape(shape), s));
      });
}
FINE_NIF(broadcast_to_nif, 0);

fine::Term concatenate_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<fine::ResourcePtr<Tensor>> arrays,
    int64_t axis) {
  return async_encoded(env, w,
      [arrays = std::move(arrays), axis](mx::Stream &s) {
        return wrap(mx::concatenate(unwrap_all(arrays), static_cast<int>(axis), s));
      });
}
FINE_NIF(concatenate_nif, 0);

fine::Term stack_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<fine::ResourcePtr<Tensor>> arrays,
    int64_t axis) {
  return async_encoded(env, w,
      [arrays = std::move(arrays), axis](mx::Stream &s) {
        return wrap(mx::stack(unwrap_all(arrays), static_cast<int>(axis), s));
      });
}
FINE_NIF(stack_nif, 0);

fine::Term flatten_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t start_axis,
    int64_t end_axis) {
  return async_encoded(env, w,
      [a = std::move(a), start_axis, end_axis](mx::Stream &s) {
        return wrap(mx::flatten(a->array, static_cast<int>(start_axis),
                                static_cast<int>(end_axis), s));
      });
}
FINE_NIF(flatten_nif, 0);

fine::Term tile_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> reps) {
  return async_encoded(env, w,
      [a = std::move(a), reps = std::move(reps)](mx::Stream &s) {
        return wrap(mx::tile(a->array, to_int_vec(reps), s));
      });
}
FINE_NIF(tile_nif, 0);

fine::Term swapaxes_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis1,
    int64_t axis2) {
  return async_encoded(env, w,
      [a = std::move(a), axis1, axis2](mx::Stream &s) {
        return wrap(mx::swapaxes(a->array, static_cast<int>(axis1),
                                 static_cast<int>(axis2), s));
      });
}
FINE_NIF(swapaxes_nif, 0);

fine::Term pad_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    std::vector<int64_t> low_pad,
    std::vector<int64_t> high_pad,
    fine::ResourcePtr<Tensor> pad_value) {
  return async_encoded(env, w,
      [a = std::move(a), axes = std::move(axes),
       low_pad = std::move(low_pad), high_pad = std::move(high_pad),
       pad_value = std::move(pad_value)](mx::Stream &s) {
        return wrap(mx::pad(a->array, to_int_vec(axes), to_mlx_shape(low_pad),
                            to_mlx_shape(high_pad), pad_value->array, "constant",
                            s));
      });
}
FINE_NIF(pad_nif, 0);

fine::Term repeat_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t repeats,
    int64_t axis) {
  return async_encoded(env, w,
      [a = std::move(a), repeats, axis](mx::Stream &s) {
        return wrap(mx::repeat(a->array, static_cast<int>(repeats),
                               static_cast<int>(axis), s));
      });
}
FINE_NIF(repeat_nif, 0);

} // namespace
