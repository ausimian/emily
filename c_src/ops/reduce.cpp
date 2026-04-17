// Reductions: sum/mean/prod/max/min/all/any (axes, keepdims);
// argmax/argmin (axis, keepdims); logsumexp; var/std (axes, keepdims, ddof).

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
using emily::wrap;

namespace {

#define EMILY_REDUCE(nif_name, mlx_fn)                                         \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *,                                                             \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      std::vector<int64_t> axes,                                               \
      bool keepdims) {                                                         \
    return w->run_sync([&](mx::Stream &s) {                                    \
      return wrap(mlx_fn(a->array, to_int_vec(axes), keepdims, s));            \
    });                                                                        \
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

#define EMILY_VARSTD(nif_name, mlx_fn)                                         \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *,                                                             \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      std::vector<int64_t> axes,                                               \
      bool keepdims,                                                           \
      int64_t ddof) {                                                          \
    return w->run_sync([&](mx::Stream &s) {                                    \
      return wrap(mlx_fn(a->array, to_int_vec(axes), keepdims,                 \
                         static_cast<int>(ddof), s));                          \
    });                                                                        \
  }                                                                            \
  FINE_NIF(nif_name, 0);

EMILY_VARSTD(var, mx::var)
EMILY_VARSTD(std, mx::std)

#undef EMILY_VARSTD

fine::ResourcePtr<Tensor> argmax(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool keepdims) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::argmax(a->array, static_cast<int>(axis), keepdims, s));
  });
}
FINE_NIF(argmax, 0);

fine::ResourcePtr<Tensor> argmin(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool keepdims) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::argmin(a->array, static_cast<int>(axis), keepdims, s));
  });
}
FINE_NIF(argmin, 0);

#define EMILY_CUM(nif_name, mlx_fn)                                            \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *,                                                             \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      int64_t axis,                                                            \
      bool reverse,                                                            \
      bool inclusive) {                                                         \
    return w->run_sync([&](mx::Stream &s) {                                    \
      return wrap(mlx_fn(a->array, static_cast<int>(axis), reverse,            \
                         inclusive, s));                                        \
    });                                                                        \
  }                                                                            \
  FINE_NIF(nif_name, 0);

EMILY_CUM(cumsum,  mx::cumsum)
EMILY_CUM(cumprod, mx::cumprod)
EMILY_CUM(cummax,  mx::cummax)
EMILY_CUM(cummin,  mx::cummin)

#undef EMILY_CUM

} // namespace
