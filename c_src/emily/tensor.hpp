// Tensor: opaque resource wrapping mlx::core::array.
//
// MLX arrays are refcounted internally; our ResourcePtr<Tensor> adds
// one BEAM-managed ref. No manual atomics, no custom destructor — fine
// and MLX together do the right thing.
//
// Helpers: wrap/unwrap shortcuts + shape conversion between Nx's
// list-of-int64 format and MLX's std::vector<int32_t> Shape.

#pragma once

#include "dtype.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace emily {

namespace mx = mlx::core;

class Tensor {
public:
  Tensor(mx::array a) : array(std::move(a)) {}
  mx::array array;
};

inline fine::ResourcePtr<Tensor> wrap(mx::array a) {
  return fine::make_resource<Tensor>(std::move(a));
}

inline mx::Shape to_mlx_shape(const std::vector<int64_t> &dims) {
  mx::Shape out;
  out.reserve(dims.size());
  for (auto d : dims) {
    if (d < 0) {
      throw std::invalid_argument("negative dimension: " + std::to_string(d));
    }
    out.push_back(static_cast<mx::ShapeElem>(d));
  }
  return out;
}

inline std::vector<int> to_int_vec(const std::vector<int64_t> &v) {
  return std::vector<int>(v.begin(), v.end());
}

inline std::vector<mx::array>
unwrap_all(const std::vector<fine::ResourcePtr<Tensor>> &tensors) {
  std::vector<mx::array> out;
  out.reserve(tensors.size());
  for (const auto &t : tensors) {
    out.push_back(t->array);
  }
  return out;
}

// Resolve a stream index from Elixir into an mx::Stream.
// -1 (the sentinel for "no explicit stream") returns the default GPU
// stream (index 0). We intentionally avoid mx::default_stream() here
// because that reads a thread-local which can be corrupted on BEAM
// scheduler threads — the BEAM migrates processes between OS threads,
// so thread-local state is unreliable.
inline mx::Stream resolve_stream(int64_t stream_index) {
  if (stream_index < 0)
    return mx::default_stream(mx::Device(mx::Device::DeviceType::gpu));
  return mx::get_stream(static_cast<int>(stream_index));
}

// MLX is not thread-safe (ml-explore/mlx#2133). In particular, the
// Metal CommandEncoder is shared state — concurrent mx::eval calls
// from different OS threads crash with "A command encoder is already
// encoding to this command buffer". BEAM dirty-CPU schedulers are a
// thread pool, so concurrent to_binary / eval NIF calls race.
//
// Serialise all mx::eval calls behind a single mutex until MLX gains
// native thread-safety (expected 0.32+, see ml-explore/mlx#3348).
inline std::mutex &eval_mutex() {
  static std::mutex m;
  return m;
}

inline void safe_eval(mx::array &a) {
  std::lock_guard<std::mutex> lock(eval_mutex());
  mx::eval(a);
}

inline void safe_eval(std::initializer_list<mx::array> arrays) {
  std::lock_guard<std::mutex> lock(eval_mutex());
  mx::eval(arrays);
}

} // namespace emily
