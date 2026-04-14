// emily_nif.cpp — minimal M0 surface: tensor round-trip.
//
// The Tensor resource wraps an mlx::core::array. MLX arrays are
// reference-counted internally; our ResourcePtr<Tensor> just adds one
// BEAM-managed ref. No manual atomics, no custom destructor — fine and
// MLX together do the right thing.

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;

// ---------- dtype mapping ----------

namespace {

mx::Dtype to_mlx_dtype(const std::string &kind, int64_t bits) {
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

std::tuple<fine::Atom, int64_t> from_mlx_dtype(mx::Dtype dtype) {
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

} // namespace

// ---------- Tensor resource ----------

class Tensor {
public:
  Tensor(mx::array a) : array(std::move(a)) {}
  mx::array array;
};

FINE_RESOURCE(Tensor);

// ---------- NIFs ----------

// from_binary/3 — build a lazy MLX array from a BEAM binary.
// Regular scheduler: MLX copies the buffer into its own storage during
// construction, so this is cheap and bounded.
fine::ResourcePtr<Tensor> from_binary(
    ErlNifEnv *,
    ErlNifBinary data,
    std::vector<int64_t> shape,
    std::tuple<fine::Atom, int64_t> dtype_tuple) {

  auto kind = std::get<0>(dtype_tuple).to_string();
  auto bits = std::get<1>(dtype_tuple);
  auto dtype = to_mlx_dtype(kind, bits);

  std::vector<int> shape_ints(shape.begin(), shape.end());

  int64_t nelem = 1;
  for (auto d : shape_ints) {
    if (d < 0) throw std::invalid_argument("negative dimension");
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
  return fine::make_resource<Tensor>(std::move(arr));
}
FINE_NIF(from_binary, 0);

// to_binary/1 — materialize the array and return its bytes as a binary.
// Dirty CPU: eval() triggers kernel launch and waits for completion.
std::string to_binary(ErlNifEnv *, fine::ResourcePtr<Tensor> tensor) {
  mx::eval(tensor->array);

  const void *src = tensor->array.data<void>();
  size_t nbytes = tensor->array.nbytes();

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
