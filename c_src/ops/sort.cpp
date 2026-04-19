// Sort / partition / topk — all along a given axis.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term sort_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis) {
  return async_encoded(env, w, [a = std::move(a), axis](mx::Stream &s) {
    return wrap(mx::sort(a->array, static_cast<int>(axis), s));
  });
}
FINE_NIF(sort_nif, 0);

fine::Term argsort_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis) {
  return async_encoded(env, w, [a = std::move(a), axis](mx::Stream &s) {
    return wrap(mx::argsort(a->array, static_cast<int>(axis), s));
  });
}
FINE_NIF(argsort_nif, 0);

fine::Term partition_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t kth,
    int64_t axis) {
  return async_encoded(env, w, [a = std::move(a), kth, axis](mx::Stream &s) {
    return wrap(mx::partition(a->array, static_cast<int>(kth),
                              static_cast<int>(axis), s));
  });
}
FINE_NIF(partition_nif, 0);

fine::Term argpartition_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t kth,
    int64_t axis) {
  return async_encoded(env, w, [a = std::move(a), kth, axis](mx::Stream &s) {
    return wrap(mx::argpartition(a->array, static_cast<int>(kth),
                                 static_cast<int>(axis), s));
  });
}
FINE_NIF(argpartition_nif, 0);

fine::Term topk_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t k,
    int64_t axis) {
  return async_encoded(env, w, [a = std::move(a), k, axis](mx::Stream &s) {
    return wrap(mx::topk(a->array, static_cast<int>(k),
                         static_cast<int>(axis), s));
  });
}
FINE_NIF(topk_nif, 0);

} // namespace
