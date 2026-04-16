// emily_nif.cpp — core NIFs: tensor resource, round-trip, eval.
//
// Op NIFs live in c_src/ops/*.cpp; they share the Tensor resource
// defined here via emily/tensor.hpp.

#include "emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::from_mlx_dtype;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;

FINE_RESOURCE(Tensor);

// ---------- Core NIFs ----------

// from_binary/3 — build a lazy MLX array from a BEAM binary.
// Regular scheduler: MLX copies the buffer into its own storage during
// construction, so this is cheap and bounded.
//
// BEAM→MLX zero-copy is not possible with the current allocator API:
// on Metal, allocator::Buffer stores an MTL::Buffer*, so wrapping a
// BEAM heap pointer would crash on GPU dispatch.
fine::ResourcePtr<Tensor> from_binary(
    ErlNifEnv *,
    ErlNifBinary data,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype_tuple) {

  auto dtype = to_mlx_dtype(dtype_tuple);
  auto shape_ints = to_mlx_shape(shape);

  int64_t nelem = 1;
  for (auto d : shape_ints) {
    nelem *= d;
  }

  size_t expected = static_cast<size_t>(nelem) * dtype.size();
  if (data.size != expected) {
    throw std::invalid_argument(
        "binary size mismatch: expected " + std::to_string(expected) +
        " got " + std::to_string(data.size));
  }

  // Allocate an MLX-owned buffer, memcpy into it, hand ownership to
  // the array with a matching deleter. See comment above for why we
  // don't alias the BEAM binary directly.
  auto buf = mx::allocator::malloc(expected);
  std::memcpy(buf.raw_ptr(), data.data, expected);
  auto deleter = [](mx::allocator::Buffer b) { mx::allocator::free(b); };

  mx::array arr(buf, std::move(shape_ints), dtype, deleter);
  return wrap(std::move(arr));
}
FINE_NIF(from_binary, 0);

// to_binary/1 — materialize the array and return its bytes as a BEAM
// resource binary aliasing MLX storage (no memcpy). The binary pins a
// Tensor resource → mx::array → MLX buffer; the buffer survives until
// the BEAM binary is GC'd.
//
// We route through mx::contiguous() so views with non-standard strides
// produce the correct in-memory layout. A handful of MLX ops (notably
// cumulative reductions on interior axes of some 4-D shapes) raise
// "Unable to safely factor shape" here; the Backend layer routes the
// known cases around us.
fine::Term to_binary(ErlNifEnv *env, fine::ResourcePtr<Tensor> tensor, int64_t s) {
  auto stream = emily::resolve_stream(s);
  auto materialized = mx::contiguous(tensor->array, false, stream);
  mx::eval(materialized);

  // Defensive: mx::contiguous is supposed to give a row-contiguous
  // layout. If it ever doesn't, we'd be aliasing a strided buffer and
  // lying about its layout. Throw rather than silently corrupt.
  if (!materialized.flags().row_contiguous) {
    throw std::runtime_error(
        "to_binary: array is not row-contiguous after mx::contiguous");
  }

  auto nbytes = materialized.nbytes();
  auto data = reinterpret_cast<const char *>(materialized.data<void>());

  auto pin = wrap(std::move(materialized));
  return fine::make_resource_binary(env, std::move(pin), data, nbytes);
}
FINE_NIF(to_binary, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// shape/1 — return the array's shape as a list of ints.
std::vector<int64_t> shape(ErlNifEnv *, fine::ResourcePtr<Tensor> tensor) {
  const auto &s = tensor->array.shape();
  return std::vector<int64_t>(s.begin(), s.end());
}
FINE_NIF(shape, 0);

// dtype/1 — return the array's dtype as an {atom, bits} tuple.
std::tuple<fine::Atom, int64_t> dtype(ErlNifEnv *, fine::ResourcePtr<Tensor> tensor) {
  return from_mlx_dtype(tensor->array.dtype());
}
FINE_NIF(dtype, 0);

// eval/1 — force evaluation of the lazy graph rooted at this tensor.
// Dirty CPU: waits for MLX to finish.
fine::Ok<> eval(ErlNifEnv *, fine::ResourcePtr<Tensor> tensor) {
  mx::eval(tensor->array);
  return fine::Ok<>{};
}
FINE_NIF(eval, ERL_NIF_DIRTY_JOB_CPU_BOUND);

FINE_INIT("Elixir.Emily.Native");
