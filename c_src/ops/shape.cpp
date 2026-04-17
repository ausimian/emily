// Shape manipulation: reshape, transpose, squeeze, expand_dims,
// broadcast_to, concatenate, stack, flatten, pad, tile, swapaxes.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::unwrap_all;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> reshape(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> shape) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::reshape(a->array, to_mlx_shape(shape), s));
  });
}
FINE_NIF(reshape, 0);

fine::ResourcePtr<Tensor> transpose(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::transpose(a->array, to_int_vec(axes), s));
  });
}
FINE_NIF(transpose, 0);

fine::ResourcePtr<Tensor> squeeze(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::squeeze(a->array, to_int_vec(axes), s));
  });
}
FINE_NIF(squeeze, 0);

fine::ResourcePtr<Tensor> expand_dims(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::expand_dims(a->array, to_int_vec(axes), s));
  });
}
FINE_NIF(expand_dims, 0);

fine::ResourcePtr<Tensor> broadcast_to(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> shape) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::broadcast_to(a->array, to_mlx_shape(shape), s));
  });
}
FINE_NIF(broadcast_to, 0);

fine::ResourcePtr<Tensor> concatenate(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<fine::ResourcePtr<Tensor>> arrays,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::concatenate(unwrap_all(arrays), static_cast<int>(axis), s));
  });
}
FINE_NIF(concatenate, 0);

fine::ResourcePtr<Tensor> stack(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<fine::ResourcePtr<Tensor>> arrays,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::stack(unwrap_all(arrays), static_cast<int>(axis), s));
  });
}
FINE_NIF(stack, 0);

fine::ResourcePtr<Tensor> flatten(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t start_axis,
    int64_t end_axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::flatten(a->array, static_cast<int>(start_axis),
                            static_cast<int>(end_axis), s));
  });
}
FINE_NIF(flatten, 0);

fine::ResourcePtr<Tensor> tile(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> reps) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::tile(a->array, to_int_vec(reps), s));
  });
}
FINE_NIF(tile, 0);

fine::ResourcePtr<Tensor> swapaxes(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis1,
    int64_t axis2) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::swapaxes(a->array, static_cast<int>(axis1),
                             static_cast<int>(axis2), s));
  });
}
FINE_NIF(swapaxes, 0);

fine::ResourcePtr<Tensor> pad(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    std::vector<int64_t> low_pad,
    std::vector<int64_t> high_pad,
    fine::ResourcePtr<Tensor> pad_value) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::pad(a->array, to_int_vec(axes), to_mlx_shape(low_pad),
                        to_mlx_shape(high_pad), pad_value->array, "constant",
                        s));
  });
}
FINE_NIF(pad, 0);

fine::ResourcePtr<Tensor> repeat(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t repeats,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::repeat(a->array, static_cast<int>(repeats),
                           static_cast<int>(axis), s));
  });
}
FINE_NIF(repeat, 0);

} // namespace
