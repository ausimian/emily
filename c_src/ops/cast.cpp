// Dtype cast.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> astype(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> a,
    std::tuple<fine::Atom, int64_t> dtype) {
  return wrap(mx::astype(a->array, to_mlx_dtype(dtype)));
}
FINE_NIF(astype, 0);

} // namespace
