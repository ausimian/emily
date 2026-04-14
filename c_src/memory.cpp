// MLX allocator introspection — exposed as NIFs so the soak harness
// can observe allocator state and assert it returns to baseline.

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>

namespace mx = mlx::core;

namespace {

int64_t get_active_memory(ErlNifEnv *) {
  return static_cast<int64_t>(mx::get_active_memory());
}
FINE_NIF(get_active_memory, 0);

int64_t get_peak_memory(ErlNifEnv *) {
  return static_cast<int64_t>(mx::get_peak_memory());
}
FINE_NIF(get_peak_memory, 0);

fine::Ok<> reset_peak_memory(ErlNifEnv *) {
  mx::reset_peak_memory();
  return fine::Ok<>{};
}
FINE_NIF(reset_peak_memory, 0);

int64_t get_cache_memory(ErlNifEnv *) {
  return static_cast<int64_t>(mx::get_cache_memory());
}
FINE_NIF(get_cache_memory, 0);

fine::Ok<> clear_cache(ErlNifEnv *) {
  mx::clear_cache();
  return fine::Ok<>{};
}
FINE_NIF(clear_cache, 0);

} // namespace
