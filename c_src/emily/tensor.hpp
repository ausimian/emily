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
// -1 (the sentinel for "no explicit stream") falls through to the
// thread-local default — backwards-compatible with pre-M14 code paths.
inline mx::Stream resolve_stream(int64_t stream_index) {
  if (stream_index < 0)
    return mx::default_stream(mx::default_device());
  return mx::get_stream(static_cast<int>(stream_index));
}

} // namespace emily
