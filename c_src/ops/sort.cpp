// Sort / partition / topk — all along a given axis.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>

namespace mx = mlx::core;
using emily::Tensor;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> sort(
    ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t axis) {
  return wrap(mx::sort(a->array, static_cast<int>(axis)));
}
FINE_NIF(sort, 0);

fine::ResourcePtr<Tensor> argsort(
    ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t axis) {
  return wrap(mx::argsort(a->array, static_cast<int>(axis)));
}
FINE_NIF(argsort, 0);

fine::ResourcePtr<Tensor> partition(
    ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t kth, int64_t axis) {
  return wrap(mx::partition(
      a->array, static_cast<int>(kth), static_cast<int>(axis)));
}
FINE_NIF(partition, 0);

fine::ResourcePtr<Tensor> argpartition(
    ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t kth, int64_t axis) {
  return wrap(mx::argpartition(
      a->array, static_cast<int>(kth), static_cast<int>(axis)));
}
FINE_NIF(argpartition, 0);

fine::ResourcePtr<Tensor> topk(
    ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t k, int64_t axis) {
  return wrap(mx::topk(a->array, static_cast<int>(k), static_cast<int>(axis)));
}
FINE_NIF(topk, 0);

} // namespace
