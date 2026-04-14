// Convolutions.
//
// Only the most general form (`conv_general`) is bound; 1-D/2-D/3-D
// specialisations can be layered in Elixir by picking strides/padding
// vectors of the right arity. This mirrors how Nx.conv/2 translates.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::to_int_vec;
using emily::wrap;

namespace {

// `padding` bundles low/high padding into a single tuple so the NIF
// arity stays manageable; both have length == spatial rank.
fine::ResourcePtr<Tensor> conv_general(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> input,
    fine::ResourcePtr<Tensor> weight,
    std::vector<int64_t> stride,
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> padding,
    std::vector<int64_t> kernel_dilation,
    std::vector<int64_t> input_dilation,
    int64_t groups,
    bool flip) {
  return wrap(mx::conv_general(
      input->array,
      weight->array,
      to_int_vec(stride),
      to_int_vec(std::get<0>(padding)),
      to_int_vec(std::get<1>(padding)),
      to_int_vec(kernel_dilation),
      to_int_vec(input_dilation),
      static_cast<int>(groups),
      flip));
}
FINE_NIF(conv_general, 0);

} // namespace
