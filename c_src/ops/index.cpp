// Indexing: slice, take, where.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
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

fine::Term slice_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> start,
    std::vector<int64_t> stop,
    std::vector<int64_t> strides) {
  return async_encoded(env, w,
      [a = std::move(a), start = std::move(start), stop = std::move(stop),
       strides = std::move(strides)](mx::Stream &s) {
        return wrap(mx::slice(a->array, to_mlx_shape(start), to_mlx_shape(stop),
                              to_mlx_shape(strides), s));
      });
}
FINE_NIF(slice_nif, 0);

fine::Term slice_update_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> src,
    fine::ResourcePtr<Tensor> update,
    std::vector<int64_t> start) {
  return async_encoded(env, w,
      [src = std::move(src), update = std::move(update),
       start = std::move(start)](mx::Stream &s) {
        const auto &update_shape = update->array.shape();
        mx::Shape start_shape = to_mlx_shape(start);
        mx::Shape stop_shape;
        stop_shape.reserve(start_shape.size());
        for (size_t i = 0; i < start_shape.size(); ++i) {
          stop_shape.push_back(start_shape[i] + update_shape[i]);
        }
        return wrap(mx::slice_update(src->array, update->array,
                                     std::move(start_shape),
                                     std::move(stop_shape), s));
      });
}
FINE_NIF(slice_update_nif, 0);

fine::Term take_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    int64_t axis) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices), axis](mx::Stream &s) {
        return wrap(mx::take(a->array, indices->array, static_cast<int>(axis), s));
      });
}
FINE_NIF(take_nif, 0);

fine::Term where_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> cond,
    fine::ResourcePtr<Tensor> x,
    fine::ResourcePtr<Tensor> y) {
  return async_encoded(env, w,
      [cond = std::move(cond), x = std::move(x),
       y = std::move(y)](mx::Stream &s) {
        return wrap(mx::where(cond->array, x->array, y->array, s));
      });
}
FINE_NIF(where_nif, 0);

fine::Term take_along_axis_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    int64_t axis) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices), axis](mx::Stream &s) {
        return wrap(mx::take_along_axis(a->array, indices->array,
                                        static_cast<int>(axis), s));
      });
}
FINE_NIF(take_along_axis_nif, 0);

fine::Term put_along_axis_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    fine::ResourcePtr<Tensor> values,
    int64_t axis) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices),
       values = std::move(values), axis](mx::Stream &s) {
        return wrap(mx::put_along_axis(a->array, indices->array, values->array,
                                       static_cast<int>(axis), s));
      });
}
FINE_NIF(put_along_axis_nif, 0);

fine::Term scatter_add_axis_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    fine::ResourcePtr<Tensor> values,
    int64_t axis) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices),
       values = std::move(values), axis](mx::Stream &s) {
        return wrap(mx::scatter_add_axis(a->array, indices->array, values->array,
                                         static_cast<int>(axis), s));
      });
}
FINE_NIF(scatter_add_axis_nif, 0);

fine::Term gather_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    std::vector<int64_t> axes,
    std::vector<int64_t> slice_sizes) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices),
       axes = std::move(axes),
       slice_sizes = std::move(slice_sizes)](mx::Stream &s) {
        return wrap(mx::gather(a->array, unwrap_all(indices), to_int_vec(axes),
                               to_mlx_shape(slice_sizes), s));
      });
}
FINE_NIF(gather_nif, 0);

fine::Term scatter_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    fine::ResourcePtr<Tensor> updates,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices),
       updates = std::move(updates), axes = std::move(axes)](mx::Stream &s) {
        return wrap(mx::scatter(a->array, unwrap_all(indices), updates->array,
                                to_int_vec(axes), s));
      });
}
FINE_NIF(scatter_nif, 0);

fine::Term scatter_add_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    fine::ResourcePtr<Tensor> updates,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), indices = std::move(indices),
       updates = std::move(updates), axes = std::move(axes)](mx::Stream &s) {
        return wrap(mx::scatter_add(a->array, unwrap_all(indices), updates->array,
                                    to_int_vec(axes), s));
      });
}
FINE_NIF(scatter_add_nif, 0);

} // namespace
