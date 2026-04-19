// Reductions: sum/mean/prod/max/min/all/any (axes, keepdims);
// argmax/argmin (axis, keepdims); logsumexp; var/std (axes, keepdims, ddof).

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
using emily::wrap;
using emily::WorkerThread;

namespace {

#define EMILY_REDUCE(op_name, mlx_fn)                                          \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      std::vector<int64_t> axes,                                               \
      bool keepdims) {                                                         \
    return async_encoded(env, w,                                               \
        [a = std::move(a), axes = std::move(axes), keepdims](mx::Stream &s) {  \
          return wrap(mlx_fn(a->array, to_int_vec(axes), keepdims, s));        \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_REDUCE(sum,       mx::sum)
EMILY_REDUCE(mean,      mx::mean)
EMILY_REDUCE(prod,      mx::prod)
EMILY_REDUCE(max,       mx::max)
EMILY_REDUCE(min,       mx::min)
EMILY_REDUCE(all,       mx::all)
EMILY_REDUCE(any,       mx::any)
EMILY_REDUCE(logsumexp, mx::logsumexp)

#undef EMILY_REDUCE

#define EMILY_VARSTD(op_name, mlx_fn)                                          \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      std::vector<int64_t> axes,                                               \
      bool keepdims,                                                           \
      int64_t ddof) {                                                          \
    return async_encoded(env, w,                                               \
        [a = std::move(a), axes = std::move(axes), keepdims,                   \
         ddof](mx::Stream &s) {                                                \
          return wrap(mlx_fn(a->array, to_int_vec(axes), keepdims,             \
                             static_cast<int>(ddof), s));                      \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_VARSTD(var, mx::var)
EMILY_VARSTD(std, mx::std)

#undef EMILY_VARSTD

fine::Term argmax_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool keepdims) {
  return async_encoded(env, w, [a = std::move(a), axis, keepdims](mx::Stream &s) {
    return wrap(mx::argmax(a->array, static_cast<int>(axis), keepdims, s));
  });
}
FINE_NIF(argmax_nif, 0);

fine::Term argmin_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool keepdims) {
  return async_encoded(env, w, [a = std::move(a), axis, keepdims](mx::Stream &s) {
    return wrap(mx::argmin(a->array, static_cast<int>(axis), keepdims, s));
  });
}
FINE_NIF(argmin_nif, 0);

#define EMILY_CUM(op_name, mlx_fn)                                             \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      int64_t axis,                                                            \
      bool reverse,                                                            \
      bool inclusive) {                                                        \
    return async_encoded(env, w,                                               \
        [a = std::move(a), axis, reverse, inclusive](mx::Stream &s) {          \
          return wrap(mlx_fn(a->array, static_cast<int>(axis), reverse,        \
                             inclusive, s));                                   \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_CUM(cumsum,  mx::cumsum)
EMILY_CUM(cumprod, mx::cumprod)
EMILY_CUM(cummax,  mx::cummax)
EMILY_CUM(cummin,  mx::cummin)

#undef EMILY_CUM

} // namespace
