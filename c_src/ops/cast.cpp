// Dtype cast.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term astype_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w, [a = std::move(a), dtype](mx::Stream &s) {
    return wrap(mx::astype(a->array, to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(astype_nif, 0);

fine::Term bitcast_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::tuple<fine::Atom, int64_t> dtype) {
  return async_encoded(env, w, [a = std::move(a), dtype](mx::Stream &s) {
    return wrap(mx::view(a->array, to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(bitcast_nif, 0);

} // namespace
