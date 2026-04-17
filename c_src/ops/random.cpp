// Random number generation.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;

namespace {

std::optional<mx::array> opt_key(
    const std::optional<fine::ResourcePtr<Tensor>> &key) {
  if (key) return (*key)->array;
  return std::nullopt;
}

fine::ResourcePtr<Tensor> random_key(ErlNifEnv *, int64_t seed) {
  return wrap(mx::random::key(static_cast<uint64_t>(seed)));
}
FINE_NIF(random_key, 0);

fine::ResourcePtr<Tensor> random_split(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> key,
    int64_t num) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::split(key->array, static_cast<int>(num), s));
  });
}
FINE_NIF(random_split, 0);

fine::ResourcePtr<Tensor> random_uniform(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> low,
    fine::ResourcePtr<Tensor> high,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::uniform(low->array, high->array,
                                    to_mlx_shape(shape), to_mlx_dtype(dtype),
                                    opt_key(key), s));
  });
}
FINE_NIF(random_uniform, 0);

fine::ResourcePtr<Tensor> random_normal(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    double loc,
    double scale,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::normal(to_mlx_shape(shape), to_mlx_dtype(dtype),
                                   static_cast<float>(loc),
                                   static_cast<float>(scale),
                                   opt_key(key), s));
  });
}
FINE_NIF(random_normal, 0);

fine::ResourcePtr<Tensor> random_randint(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> low,
    fine::ResourcePtr<Tensor> high,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::randint(low->array, high->array,
                                    to_mlx_shape(shape), to_mlx_dtype(dtype),
                                    opt_key(key), s));
  });
}
FINE_NIF(random_randint, 0);

fine::ResourcePtr<Tensor> random_bernoulli(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> p,
    std::vector<int64_t> shape,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::bernoulli(p->array, to_mlx_shape(shape),
                                      opt_key(key), s));
  });
}
FINE_NIF(random_bernoulli, 0);

fine::ResourcePtr<Tensor> random_gumbel(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::gumbel(to_mlx_shape(shape), to_mlx_dtype(dtype),
                                   opt_key(key), s));
  });
}
FINE_NIF(random_gumbel, 0);

fine::ResourcePtr<Tensor> random_categorical(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> logits,
    int64_t axis,
    int64_t num_samples,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::random::categorical(logits->array, static_cast<int>(axis),
                                        static_cast<int>(num_samples),
                                        opt_key(key), s));
  });
}
FINE_NIF(random_categorical, 0);

} // namespace
