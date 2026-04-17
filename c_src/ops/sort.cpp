// Sort / partition / topk — all along a given axis.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> sort(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::sort(a->array, static_cast<int>(axis), s));
  });
}
FINE_NIF(sort, 0);

fine::ResourcePtr<Tensor> argsort(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::argsort(a->array, static_cast<int>(axis), s));
  });
}
FINE_NIF(argsort, 0);

fine::ResourcePtr<Tensor> partition(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t kth,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::partition(a->array, static_cast<int>(kth),
                              static_cast<int>(axis), s));
  });
}
FINE_NIF(partition, 0);

fine::ResourcePtr<Tensor> argpartition(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t kth,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::argpartition(a->array, static_cast<int>(kth),
                                 static_cast<int>(axis), s));
  });
}
FINE_NIF(argpartition, 0);

fine::ResourcePtr<Tensor> topk(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t k,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::topk(a->array, static_cast<int>(k),
                         static_cast<int>(axis), s));
  });
}
FINE_NIF(topk, 0);

} // namespace
