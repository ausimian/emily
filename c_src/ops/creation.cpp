// Creation ops: zeros, ones, full, arange, eye.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term zeros_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w, [shape = std::move(shape), dtype](mx::Stream &s) {
    return wrap(mx::zeros(to_mlx_shape(shape), to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(zeros_nif, 0);

fine::Term ones_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w, [shape = std::move(shape), dtype](mx::Stream &s) {
    return wrap(mx::ones(to_mlx_shape(shape), to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(ones_nif, 0);

fine::Term full_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    fine::ResourcePtr<Tensor> value,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w,
      [shape = std::move(shape), value = std::move(value), dtype](mx::Stream &s) {
        return wrap(mx::full(to_mlx_shape(shape), value->array,
                             to_mlx_dtype(dtype), s));
      });
}
FINE_NIF(full_nif, 0);

fine::Term arange_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    double start,
    double stop,
    double step,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w, [start, stop, step, dtype](mx::Stream &s) {
    return wrap(mx::arange(start, stop, step, to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(arange_nif, 0);

fine::Term eye_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    int64_t n,
    int64_t m,
    int64_t k,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w, [n, m, k, dtype](mx::Stream &s) {
    return wrap(mx::eye(static_cast<int>(n), static_cast<int>(m),
                        static_cast<int>(k), to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(eye_nif, 0);

} // namespace
