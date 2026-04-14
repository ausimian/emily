// Dtype translation between Nx's {kind, bits} tuples and mlx::Dtype.
//
// Nx kinds we honour: "f" (float), "bf" (bfloat), "s" (signed int),
// "u" (unsigned int), "c" (complex), "pred" (1-bit bool, MLX stores
// as one byte).

#pragma once

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>

namespace emily {

namespace mx = mlx::core;

inline mx::Dtype to_mlx_dtype(const std::string &kind, int64_t bits) {
  if (kind == "f"  && bits == 32) return mx::float32;
  if (kind == "f"  && bits == 16) return mx::float16;
  if (kind == "bf" && bits == 16) return mx::bfloat16;
  if (kind == "s"  && bits ==  8) return mx::int8;
  if (kind == "s"  && bits == 16) return mx::int16;
  if (kind == "s"  && bits == 32) return mx::int32;
  if (kind == "s"  && bits == 64) return mx::int64;
  if (kind == "u"  && bits ==  8) return mx::uint8;
  if (kind == "u"  && bits == 16) return mx::uint16;
  if (kind == "u"  && bits == 32) return mx::uint32;
  if (kind == "u"  && bits == 64) return mx::uint64;
  if (kind == "c"  && bits == 64) return mx::complex64;
  if (kind == "pred")             return mx::bool_;

  throw std::invalid_argument(
      "unsupported dtype: {" + kind + ", " + std::to_string(bits) + "}");
}

inline mx::Dtype to_mlx_dtype(const std::tuple<fine::Atom, int64_t> &t) {
  return to_mlx_dtype(std::get<0>(t).to_string(), std::get<1>(t));
}

inline std::tuple<fine::Atom, int64_t> from_mlx_dtype(mx::Dtype dtype) {
  if (dtype == mx::float32)   return {fine::Atom("f"),    32};
  if (dtype == mx::float16)   return {fine::Atom("f"),    16};
  if (dtype == mx::bfloat16)  return {fine::Atom("bf"),   16};
  if (dtype == mx::int8)      return {fine::Atom("s"),     8};
  if (dtype == mx::int16)     return {fine::Atom("s"),    16};
  if (dtype == mx::int32)     return {fine::Atom("s"),    32};
  if (dtype == mx::int64)     return {fine::Atom("s"),    64};
  if (dtype == mx::uint8)     return {fine::Atom("u"),     8};
  if (dtype == mx::uint16)    return {fine::Atom("u"),    16};
  if (dtype == mx::uint32)    return {fine::Atom("u"),    32};
  if (dtype == mx::uint64)    return {fine::Atom("u"),    64};
  if (dtype == mx::complex64) return {fine::Atom("c"),    64};
  if (dtype == mx::bool_)     return {fine::Atom("pred"),  1};
  throw std::runtime_error("unmapped mlx dtype");
}

} // namespace emily
