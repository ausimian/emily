// Indexing: slice, take, where.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::unwrap_all;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> slice(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> start,
    std::vector<int64_t> stop,
    std::vector<int64_t> strides) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::slice(a->array, to_mlx_shape(start), to_mlx_shape(stop),
                          to_mlx_shape(strides), s));
  });
}
FINE_NIF(slice, 0);

fine::ResourcePtr<Tensor> slice_update(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> src,
    fine::ResourcePtr<Tensor> update,
    std::vector<int64_t> start) {
  return w->run_sync([&](mx::Stream &s) {
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
FINE_NIF(slice_update, 0);

fine::ResourcePtr<Tensor> take(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::take(a->array, indices->array, static_cast<int>(axis), s));
  });
}
FINE_NIF(take, 0);

fine::ResourcePtr<Tensor> where(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> cond,
    fine::ResourcePtr<Tensor> x,
    fine::ResourcePtr<Tensor> y) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::where(cond->array, x->array, y->array, s));
  });
}
FINE_NIF(where, 0);

fine::ResourcePtr<Tensor> take_along_axis(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::take_along_axis(a->array, indices->array,
                                    static_cast<int>(axis), s));
  });
}
FINE_NIF(take_along_axis, 0);

fine::ResourcePtr<Tensor> put_along_axis(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    fine::ResourcePtr<Tensor> values,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::put_along_axis(a->array, indices->array, values->array,
                                   static_cast<int>(axis), s));
  });
}
FINE_NIF(put_along_axis, 0);

fine::ResourcePtr<Tensor> scatter_add_axis(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    fine::ResourcePtr<Tensor> values,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::scatter_add_axis(a->array, indices->array, values->array,
                                     static_cast<int>(axis), s));
  });
}
FINE_NIF(scatter_add_axis, 0);

fine::ResourcePtr<Tensor> gather(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    std::vector<int64_t> axes,
    std::vector<int64_t> slice_sizes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::gather(a->array, unwrap_all(indices), to_int_vec(axes),
                           to_mlx_shape(slice_sizes), s));
  });
}
FINE_NIF(gather, 0);

fine::ResourcePtr<Tensor> scatter(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    fine::ResourcePtr<Tensor> updates,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::scatter(a->array, unwrap_all(indices), updates->array,
                            to_int_vec(axes), s));
  });
}
FINE_NIF(scatter, 0);

fine::ResourcePtr<Tensor> scatter_add(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    fine::ResourcePtr<Tensor> updates,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::scatter_add(a->array, unwrap_all(indices), updates->array,
                                to_int_vec(axes), s));
  });
}
FINE_NIF(scatter_add, 0);

} // namespace
