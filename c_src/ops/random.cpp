// Random number generation.
//
// Keys are passed as `std::optional<Tensor>`; nil from Elixir uses
// MLX's default key sequence. Explicit keys let callers make a
// computation deterministic.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;

namespace {

// Unwrap an optional<ResourcePtr<Tensor>> into an optional<mx::array>.
std::optional<mx::array> opt_key(
    const std::optional<fine::ResourcePtr<Tensor>> &key) {
  if (key) return (*key)->array;
  return std::nullopt;
}

// random_key/1 — build a PRNG key tensor from an integer seed.
fine::ResourcePtr<Tensor> random_key(ErlNifEnv *, int64_t seed) {
  return wrap(mx::random::key(static_cast<uint64_t>(seed)));
}
FINE_NIF(random_key, 0);

// random_split/2 — split a key into `num` new keys.
fine::ResourcePtr<Tensor> random_split(
    ErlNifEnv *, fine::ResourcePtr<Tensor> key, int64_t num) {
  return wrap(mx::random::split(key->array, static_cast<int>(num)));
}
FINE_NIF(random_split, 0);

// random_uniform/5 — uniform(low, high, shape, dtype, key?).
fine::ResourcePtr<Tensor> random_uniform(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> low,
    fine::ResourcePtr<Tensor> high,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return wrap(mx::random::uniform(
      low->array,
      high->array,
      to_mlx_shape(shape),
      to_mlx_dtype(dtype),
      opt_key(key)));
}
FINE_NIF(random_uniform, 0);

// random_normal/5 — normal(shape, dtype, loc, scale, key?).
fine::ResourcePtr<Tensor> random_normal(
    ErlNifEnv *,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    double loc,
    double scale,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return wrap(mx::random::normal(
      to_mlx_shape(shape),
      to_mlx_dtype(dtype),
      static_cast<float>(loc),
      static_cast<float>(scale),
      opt_key(key)));
}
FINE_NIF(random_normal, 0);

// random_randint/5 — uniform integers in [low, high).
fine::ResourcePtr<Tensor> random_randint(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> low,
    fine::ResourcePtr<Tensor> high,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return wrap(mx::random::randint(
      low->array,
      high->array,
      to_mlx_shape(shape),
      to_mlx_dtype(dtype),
      opt_key(key)));
}
FINE_NIF(random_randint, 0);

// random_bernoulli/3 — bernoulli(p, shape, key?).
fine::ResourcePtr<Tensor> random_bernoulli(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> p,
    std::vector<int64_t> shape,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return wrap(mx::random::bernoulli(
      p->array, to_mlx_shape(shape), opt_key(key)));
}
FINE_NIF(random_bernoulli, 0);

// random_gumbel/3 — gumbel(shape, dtype, key?).
fine::ResourcePtr<Tensor> random_gumbel(
    ErlNifEnv *,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return wrap(mx::random::gumbel(
      to_mlx_shape(shape), to_mlx_dtype(dtype), opt_key(key)));
}
FINE_NIF(random_gumbel, 0);

// random_categorical/4 — sample class indices from logits.
fine::ResourcePtr<Tensor> random_categorical(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> logits,
    int64_t axis,
    int64_t num_samples,
    std::optional<fine::ResourcePtr<Tensor>> key) {
  return wrap(mx::random::categorical(
      logits->array,
      static_cast<int>(axis),
      static_cast<int>(num_samples),
      opt_key(key)));
}
FINE_NIF(random_categorical, 0);

} // namespace
