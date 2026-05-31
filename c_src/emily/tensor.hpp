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

#include <cstddef>
#include <cstdint>
#include <limits>
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
    // MLX's ShapeElem is int32_t. Without this bound a dimension above
    // INT32_MAX would silently truncate (e.g. 2^31 -> INT32_MIN), giving
    // an array whose recorded shape doesn't match its buffer.
    if (d > std::numeric_limits<mx::ShapeElem>::max()) {
      throw std::invalid_argument("dimension exceeds int32 max: " +
                                  std::to_string(d));
    }
    out.push_back(static_cast<mx::ShapeElem>(d));
  }
  return out;
}

// Element count of a shape, computed in size_t with overflow checking.
// Dimensions must already be validated nonnegative (see to_mlx_shape).
// Throws if the running product overflows size_t — without this guard
// the product can wrap (e.g. [2^21, 2^21, 2^22] wraps to 0) and let an
// undersized binary pass a byte-size check, building an array whose
// shape outruns its allocation.
inline std::size_t checked_nelem(const mx::Shape &shape) {
  std::size_t nelem = 1;
  for (auto d : shape) {
    if (__builtin_mul_overflow(nelem, static_cast<std::size_t>(d), &nelem)) {
      throw std::invalid_argument("element count overflow");
    }
  }
  return nelem;
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

} // namespace emily
