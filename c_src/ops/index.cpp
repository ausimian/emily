// Indexing: slice, take, where.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_mlx_shape;
using emily::wrap;

namespace {

// slice/4: a[start:stop:strides] per-axis. All three vectors have
// length == rank(a).
fine::ResourcePtr<Tensor> slice(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> start,
    std::vector<int64_t> stop,
    std::vector<int64_t> strides) {
  return wrap(mx::slice(
      a->array,
      to_mlx_shape(start),
      to_mlx_shape(stop),
      to_mlx_shape(strides)));
}
FINE_NIF(slice, 0);

// slice_update/3: write `update` into `src` starting at `start`. `stop`
// is derived as `start + shape(update)` and strides default to 1 on
// every axis (Nx.put_slice has no stride parameter). Output shape
// equals `src.shape`.
fine::ResourcePtr<Tensor> slice_update(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> src,
    fine::ResourcePtr<Tensor> update,
    std::vector<int64_t> start) {
  const auto &update_shape = update->array.shape();
  mx::Shape start_shape = to_mlx_shape(start);
  mx::Shape stop_shape;
  stop_shape.reserve(start_shape.size());
  for (size_t i = 0; i < start_shape.size(); ++i) {
    stop_shape.push_back(start_shape[i] + update_shape[i]);
  }
  return wrap(mx::slice_update(
      src->array, update->array, std::move(start_shape), std::move(stop_shape)));
}
FINE_NIF(slice_update, 0);

// take/3: gather along `axis` using integer indices.
fine::ResourcePtr<Tensor> take(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    int64_t axis) {
  return wrap(mx::take(a->array, indices->array, static_cast<int>(axis)));
}
FINE_NIF(take, 0);

fine::ResourcePtr<Tensor> where(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> cond,
    fine::ResourcePtr<Tensor> x,
    fine::ResourcePtr<Tensor> y) {
  return wrap(mx::where(cond->array, x->array, y->array));
}
FINE_NIF(where, 0);

// take_along_axis/3 — gather along `axis` using integer indices whose
// shape matches `a` except along `axis`.
fine::ResourcePtr<Tensor> take_along_axis(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    int64_t axis) {
  return wrap(mx::take_along_axis(
      a->array, indices->array, static_cast<int>(axis)));
}
FINE_NIF(take_along_axis, 0);

// put_along_axis/4 — write `values` into `a` at the given indices
// along `axis`.
fine::ResourcePtr<Tensor> put_along_axis(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    fine::ResourcePtr<Tensor> values,
    int64_t axis) {
  return wrap(mx::put_along_axis(
      a->array, indices->array, values->array, static_cast<int>(axis)));
}
FINE_NIF(put_along_axis, 0);

// scatter_add_axis/4 — add `values` into `a` at the given indices
// along `axis`.
fine::ResourcePtr<Tensor> scatter_add_axis(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> indices,
    fine::ResourcePtr<Tensor> values,
    int64_t axis) {
  return wrap(mx::scatter_add_axis(
      a->array, indices->array, values->array, static_cast<int>(axis)));
}
FINE_NIF(scatter_add_axis, 0);

} // namespace
