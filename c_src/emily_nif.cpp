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

  // MLX has no void*-accepting array constructor. The canonical path
  // is: allocate an MLX-owned buffer, memcpy into it, hand ownership
  // to the array with a matching deleter.
  auto buf = mx::allocator::malloc(expected);
  std::memcpy(buf.raw_ptr(), data.data, expected);
  auto deleter = [](mx::allocator::Buffer b) { mx::allocator::free(b); };

  mx::array arr(buf, std::move(shape_ints), dtype, deleter);
  return wrap(std::move(arr));
}
FINE_NIF(from_binary, 0);

// to_binary/1 — materialize the array and return its bytes as a binary.
// Dirty CPU: eval() triggers kernel launch and waits for completion.
//
// We route through mx::contiguous() first so views with non-standard
// strides (transpose, slice, swapaxes, broadcast_to) produce the
// correct in-memory layout. For already-contiguous arrays MLX elides
// the copy.
std::string to_binary(ErlNifEnv *, fine::ResourcePtr<Tensor> tensor) {
  auto materialized = mx::contiguous(tensor->array);
  mx::eval(materialized);

  const void *src = materialized.data<void>();
  size_t nbytes = materialized.nbytes();

  std::string out;
  out.resize(nbytes);
  std::memcpy(out.data(), src, nbytes);
  return out;
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
