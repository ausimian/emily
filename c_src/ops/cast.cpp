// Dtype cast.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_mlx_dtype;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> astype(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::tuple<fine::Atom, int64_t> dtype) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::astype(a->array, to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(astype, 0);

fine::ResourcePtr<Tensor> bitcast(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::tuple<fine::Atom, int64_t> dtype) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::view(a->array, to_mlx_dtype(dtype), s));
  });
}
FINE_NIF(bitcast, 0);

} // namespace
