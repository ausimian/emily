// MLX stream management — exposed as NIFs so each BEAM process can
// get its own Metal command queue for concurrent inference.

#include "emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace mx = mlx::core;

namespace {

mx::Device::DeviceType to_device_type(fine::Atom device_atom) {
  auto name = device_atom.to_string();
  if (name == "gpu") return mx::Device::DeviceType::gpu;
  if (name == "cpu") return mx::Device::DeviceType::cpu;
  throw std::invalid_argument(
      "device must be :gpu or :cpu, got: " + name);
}

// new_stream/1 — create a new Metal command queue on the given device.
// Returns the stream index (integer).
int64_t new_stream(ErlNifEnv *, fine::Atom device_atom) {
  auto stream = mx::new_stream(mx::Device(to_device_type(device_atom)));
  return static_cast<int64_t>(stream.index);
}
FINE_NIF(new_stream, 0);

// set_default_stream/1 — set the thread-local default stream.
//
// WARNING: BEAM processes migrate between OS threads, so thread-local
// state is unreliable. with_stream/2 no longer calls this — it routes
// streams via the process dictionary instead. This NIF is retained for
// advanced use cases but should be avoided in normal code.
fine::Ok<> set_default_stream(ErlNifEnv *, int64_t stream_index) {
  mx::set_default_stream(mx::get_stream(static_cast<int>(stream_index)));
  return fine::Ok<>{};
}
FINE_NIF(set_default_stream, 0);

// get_default_stream/1 — return the index of the default stream for
// a device.
int64_t get_default_stream(ErlNifEnv *, fine::Atom device_atom) {
  auto stream = mx::default_stream(mx::Device(to_device_type(device_atom)));
  return static_cast<int64_t>(stream.index);
}
FINE_NIF(get_default_stream, 0);

// synchronize_stream/1 — block until all ops on the stream complete.
fine::Ok<> synchronize_stream(ErlNifEnv *, int64_t stream_index) {
  mx::synchronize(mx::get_stream(static_cast<int>(stream_index)));
  return fine::Ok<>{};
}
FINE_NIF(synchronize_stream, ERL_NIF_DIRTY_JOB_CPU_BOUND);

} // namespace
