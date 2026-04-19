// Random number generation.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;
using emily::WorkerThread;

namespace {

std::optional<mx::array> opt_key(
    const std::optional<fine::ResourcePtr<Tensor>> &key) {
  if (key) return (*key)->array;
  return std::nullopt;
}

// random_key/1 is pure (no stream, no worker needed) — stays sync.
fine::ResourcePtr<Tensor> random_key(ErlNifEnv *, int64_t seed) {
  return wrap(mx::random::key(static_cast<uint64_t>(seed)));
}
FINE_NIF(random_key, 0);

fine::Term random_split_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> key,
    int64_t num) {
  return async_encoded(env, w, [key = std::move(key), num](mx::Stream &s) {
    return wrap(mx::random::split(key->array, static_cast<int>(num), s));
  });
}
FINE_NIF(random_split_nif, 0);

fine::Term random_uniform_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> low,
    fine::ResourcePtr<Tensor> high,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return async_encoded(env, w,
      [low = std::move(low), high = std::move(high),
       shape = std::move(shape), dtype,
       key = std::move(key)](mx::Stream &s) {
        return wrap(mx::random::uniform(low->array, high->array,
                                        to_mlx_shape(shape), to_mlx_dtype(dtype),
                                        opt_key(key), s));
      });
}
FINE_NIF(random_uniform_nif, 0);

fine::Term random_normal_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    double loc,
    double scale,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return async_encoded(env, w,
      [shape = std::move(shape), dtype, loc, scale,
       key = std::move(key)](mx::Stream &s) {
        return wrap(mx::random::normal(to_mlx_shape(shape), to_mlx_dtype(dtype),
                                       static_cast<float>(loc),
                                       static_cast<float>(scale),
                                       opt_key(key), s));
      });
}
FINE_NIF(random_normal_nif, 0);

fine::Term random_randint_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> low,
    fine::ResourcePtr<Tensor> high,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return async_encoded(env, w,
      [low = std::move(low), high = std::move(high),
       shape = std::move(shape), dtype,
       key = std::move(key)](mx::Stream &s) {
        return wrap(mx::random::randint(low->array, high->array,
                                        to_mlx_shape(shape), to_mlx_dtype(dtype),
                                        opt_key(key), s));
      });
}
FINE_NIF(random_randint_nif, 0);

fine::Term random_bernoulli_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> p,
    std::vector<int64_t> shape,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return async_encoded(env, w,
      [p = std::move(p), shape = std::move(shape),
       key = std::move(key)](mx::Stream &s) {
        return wrap(mx::random::bernoulli(p->array, to_mlx_shape(shape),
                                          opt_key(key), s));
      });
}
FINE_NIF(random_bernoulli_nif, 0);

fine::Term random_gumbel_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return async_encoded(env, w,
      [shape = std::move(shape), dtype,
       key = std::move(key)](mx::Stream &s) {
        return wrap(mx::random::gumbel(to_mlx_shape(shape), to_mlx_dtype(dtype),
                                       opt_key(key), s));
      });
}
FINE_NIF(random_gumbel_nif, 0);

fine::Term random_categorical_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> logits,
    int64_t axis,
    int64_t num_samples,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return async_encoded(env, w,
      [logits = std::move(logits), axis, num_samples,
       key = std::move(key)](mx::Stream &s) {
        return wrap(mx::random::categorical(logits->array, static_cast<int>(axis),
                                            static_cast<int>(num_samples),
                                            opt_key(key), s));
      });
}
FINE_NIF(random_categorical_nif, 0);

} // namespace
