// Creation ops: zeros, ones, full, arange, eye.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> zeros(
    ErlNifEnv *,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype) {
  return wrap(mx::zeros(to_mlx_shape(shape), to_mlx_dtype(dtype)));
}
FINE_NIF(zeros, 0);

fine::ResourcePtr<Tensor> ones(
    ErlNifEnv *,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype) {
  return wrap(mx::ones(to_mlx_shape(shape), to_mlx_dtype(dtype)));
}
FINE_NIF(ones, 0);

// full/3: broadcasts `value` (any shape, typically scalar) to `shape`,
// cast to `dtype`.
fine::ResourcePtr<Tensor> full(
    ErlNifEnv *,
    std::vector<int64_t> shape,
    fine::ResourcePtr<Tensor> value,
    std::tuple<fine::Atom, int64_t> dtype) {
  return wrap(mx::full(to_mlx_shape(shape), value->array, to_mlx_dtype(dtype)));
}
FINE_NIF(full, 0);

// arange/4: mirrors mlx::arange(start, stop, step, dtype). Ints are
// widened to doubles at the boundary; MLX handles the dtype cast.
fine::ResourcePtr<Tensor> arange(
    ErlNifEnv *,
    double start,
    double stop,
    double step,
    std::tuple<fine::Atom, int64_t> dtype) {
  return wrap(mx::arange(start, stop, step, to_mlx_dtype(dtype)));
}
FINE_NIF(arange, 0);

// eye/4: n×m with ones on diagonal k.
fine::ResourcePtr<Tensor> eye(
    ErlNifEnv *,
    int64_t n,
    int64_t m,
    int64_t k,
    std::tuple<fine::Atom, int64_t> dtype) {
  return wrap(mx::eye(static_cast<int>(n),
                      static_cast<int>(m),
                      static_cast<int>(k),
                      to_mlx_dtype(dtype)));
}
FINE_NIF(eye, 0);

} // namespace
