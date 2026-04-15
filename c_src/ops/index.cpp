// Indexing: slice, take, where.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::unwrap_all;
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

// gather/4 — multi-axis fancy-indexing gather. `indices` is a list of
// integer index tensors (one per axis in `axes`); all index tensors
// share a common leading batch shape. `slice_sizes` has length
// rank(a); entries for the indexed axes are 1 and entries for the
// remaining axes equal a.shape()[axis] (the window size per gather).
// Result shape is `batch_shape ++ slice_sizes`.
fine::ResourcePtr<Tensor> gather(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    std::vector<int64_t> axes,
    std::vector<int64_t> slice_sizes) {
  return wrap(mx::gather(
      a->array,
      unwrap_all(indices),
      to_int_vec(axes),
      to_mlx_shape(slice_sizes)));
}
FINE_NIF(gather, 0);

// scatter/4 — multi-axis fancy-indexing scatter (overwrite). MLX
// requires `updates.ndim() == indices[0].ndim() + a.ndim()`; each
// update is a slice of a.ndim() dims written at the index site. The
// Backend layer is responsible for reshaping Nx-shaped updates into
// MLX's expected shape.
//
// Note: MLX scatter with duplicate indices has unordered semantics
// (parallel write) — it is not last-write-wins like Nx.indexed_put.
// Callers requiring deterministic duplicate handling must dedupe
// beforehand.
fine::ResourcePtr<Tensor> scatter(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    fine::ResourcePtr<Tensor> updates,
    std::vector<int64_t> axes) {
  return wrap(mx::scatter(
      a->array, unwrap_all(indices), updates->array, to_int_vec(axes)));
}
FINE_NIF(scatter, 0);

// scatter_add/4 — multi-axis scatter-accumulate. Same shape contract
// as scatter; duplicate indices accumulate deterministically (add is
// commutative, so parallel scatter order doesn't affect the result).
fine::ResourcePtr<Tensor> scatter_add(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::vector<fine::ResourcePtr<Tensor>> indices,
    fine::ResourcePtr<Tensor> updates,
    std::vector<int64_t> axes) {
  return wrap(mx::scatter_add(
      a->array, unwrap_all(indices), updates->array, to_int_vec(axes)));
}
FINE_NIF(scatter_add, 0);

} // namespace
