// Fused transformer kernels from mlx::core::fast.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/fast.h>
#include <mlx/mlx.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::unwrap_all;
using emily::wrap;

namespace {

std::optional<mx::array> opt_array(
    const std::optional<fine::ResourcePtr<Tensor>> &opt) {
  if (opt) return (*opt)->array;
  return std::nullopt;
}

fine::ResourcePtr<Tensor> fast_rms_norm(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    std::optional<fine::ResourcePtr<Tensor>> weight,
    double eps) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::fast::rms_norm(x->array, opt_array(weight),
                                   static_cast<float>(eps), s));
  });
}
FINE_NIF(fast_rms_norm, 0);

fine::ResourcePtr<Tensor> fast_layer_norm(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    std::optional<fine::ResourcePtr<Tensor>> weight,
    std::optional<fine::ResourcePtr<Tensor>> bias,
    double eps) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::fast::layer_norm(x->array, opt_array(weight),
                                     opt_array(bias),
                                     static_cast<float>(eps), s));
  });
}
FINE_NIF(fast_layer_norm, 0);

// `offset` is always a tensor (Bumblebee tracks cumulative position as
// Nx.Tensor through iterative decode); uses the array-offset overload.
fine::ResourcePtr<Tensor> fast_rope(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    int64_t dims,
    bool traditional,
    std::optional<double> base,
    double scale,
    fine::ResourcePtr<Tensor> offset,
    std::optional<fine::ResourcePtr<Tensor>> freqs) {
  return w->run_sync([&](mx::Stream &s) {
    std::optional<float> base_f;
    if (base) base_f = static_cast<float>(*base);
    return wrap(mx::fast::rope(x->array, static_cast<int>(dims), traditional,
                               base_f, static_cast<float>(scale),
                               offset->array, opt_array(freqs), s));
  });
}
FINE_NIF(fast_rope, 0);

fine::ResourcePtr<Tensor> fast_scaled_dot_product_attention(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> q,
    fine::ResourcePtr<Tensor> k,
    fine::ResourcePtr<Tensor> v,
    double scale,
    std::string mask_mode,
    std::vector<fine::ResourcePtr<Tensor>> mask_arrs) {
  return w->run_sync([&](mx::Stream &s) {
    std::optional<mx::array> mask_arr;
    if (!mask_arrs.empty()) {
      mask_arr = mask_arrs[0]->array;
    }
    return wrap(mx::fast::scaled_dot_product_attention(
        q->array, k->array, v->array, static_cast<float>(scale), mask_mode,
        mask_arr, std::nullopt, s));
  });
}
FINE_NIF(fast_scaled_dot_product_attention, 0);

} // namespace
