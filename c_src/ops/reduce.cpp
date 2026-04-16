// Reductions: sum/mean/prod/max/min/all/any (axes, keepdims);
// argmax/argmin (axis, keepdims); logsumexp; var/std (axes, keepdims, ddof).

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

#define EMILY_REDUCE(nif_name, mlx_fn)                                         \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *,                                                             \
      fine::ResourcePtr<Tensor> a,                                             \
      std::vector<int64_t> axes,                                               \
      bool keepdims,                                                           \
      int64_t s) {                                                             \
    return wrap(                                                               \
        mlx_fn(a->array, to_int_vec(axes), keepdims,                           \
               emily::resolve_stream(s)));                                     \
  }                                                                            \
  FINE_NIF(nif_name, 0);

EMILY_REDUCE(sum,       mx::sum)
EMILY_REDUCE(mean,      mx::mean)
EMILY_REDUCE(prod,      mx::prod)
EMILY_REDUCE(max,       mx::max)
EMILY_REDUCE(min,       mx::min)
EMILY_REDUCE(all,       mx::all)
EMILY_REDUCE(any,       mx::any)
EMILY_REDUCE(logsumexp, mx::logsumexp)

#undef EMILY_REDUCE

// var/std take an extra ddof parameter.
#define EMILY_VARSTD(nif_name, mlx_fn)                                         \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *,                                                             \
      fine::ResourcePtr<Tensor> a,                                             \
      std::vector<int64_t> axes,                                               \
      bool keepdims,                                                           \
      int64_t ddof,                                                            \
      int64_t s) {                                                             \
    return wrap(mlx_fn(a->array, to_int_vec(axes), keepdims,                   \
                       static_cast<int>(ddof),                                 \
                       emily::resolve_stream(s)));                             \
  }                                                                            \
  FINE_NIF(nif_name, 0);

EMILY_VARSTD(var, mx::var)
EMILY_VARSTD(std, mx::std)

#undef EMILY_VARSTD

// argmax / argmin reduce a single axis.
fine::ResourcePtr<Tensor> argmax(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool keepdims,
    int64_t s) {
  return wrap(mx::argmax(a->array, static_cast<int>(axis), keepdims,
                         emily::resolve_stream(s)));
}
FINE_NIF(argmax, 0);

fine::ResourcePtr<Tensor> argmin(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool keepdims,
    int64_t s) {
  return wrap(mx::argmin(a->array, static_cast<int>(axis), keepdims,
                         emily::resolve_stream(s)));
}
FINE_NIF(argmin, 0);

// cumulative reductions: axis, reverse, inclusive.
#define EMILY_CUM(nif_name, mlx_fn)                                            \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *,                                                             \
      fine::ResourcePtr<Tensor> a,                                             \
      int64_t axis,                                                            \
      bool reverse,                                                            \
      bool inclusive,                                                          \
      int64_t s) {                                                             \
    return wrap(mlx_fn(a->array, static_cast<int>(axis), reverse, inclusive,   \
                       emily::resolve_stream(s)));                             \
  }                                                                            \
  FINE_NIF(nif_name, 0);

EMILY_CUM(cumsum,  mx::cumsum)
EMILY_CUM(cumprod, mx::cumprod)
EMILY_CUM(cummax,  mx::cummax)
EMILY_CUM(cummin,  mx::cummin)

#undef EMILY_CUM

} // namespace
