// Fused transformer kernels from mlx::core::fast.

#include "../emily/async.hpp"
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
using emily::async_encoded;
using emily::Tensor;
using emily::unwrap_all;
using emily::wrap;
using emily::WorkerThread;

namespace {

std::optional<mx::array> opt_array(
    const std::optional<fine::ResourcePtr<Tensor>> &opt) {
  if (opt) return (*opt)->array;
  return std::nullopt;
}

fine::Term fast_rms_norm_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    std::optional<fine::ResourcePtr<Tensor>> weight,
    double eps) {
  return async_encoded(env, w,
      [x = std::move(x), weight = std::move(weight), eps](mx::Stream &s) {
        return wrap(mx::fast::rms_norm(x->array, opt_array(weight),
                                       static_cast<float>(eps), s));
      });
}
FINE_NIF(fast_rms_norm_nif, 0);

fine::Term fast_layer_norm_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    std::optional<fine::ResourcePtr<Tensor>> weight,
    std::optional<fine::ResourcePtr<Tensor>> bias,
    double eps) {
  return async_encoded(env, w,
      [x = std::move(x), weight = std::move(weight),
       bias = std::move(bias), eps](mx::Stream &s) {
        return wrap(mx::fast::layer_norm(x->array, opt_array(weight),
                                         opt_array(bias),
                                         static_cast<float>(eps), s));
      });
}
FINE_NIF(fast_layer_norm_nif, 0);

// `offset` is always a tensor (Bumblebee tracks cumulative position as
// Nx.Tensor through iterative decode); uses the array-offset overload.
fine::Term fast_rope_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    int64_t dims,
    bool traditional,
    std::optional<double> base,
    double scale,
    fine::ResourcePtr<Tensor> offset,
    std::optional<fine::ResourcePtr<Tensor>> freqs) {
  return async_encoded(env, w,
      [x = std::move(x), dims, traditional, base, scale,
       offset = std::move(offset), freqs = std::move(freqs)](mx::Stream &s) {
        std::optional<float> base_f;
        if (base) base_f = static_cast<float>(*base);
        return wrap(mx::fast::rope(x->array, static_cast<int>(dims), traditional,
                                   base_f, static_cast<float>(scale),
                                   offset->array, opt_array(freqs), s));
      });
}
FINE_NIF(fast_rope_nif, 0);

fine::Term fast_scaled_dot_product_attention_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> q,
    fine::ResourcePtr<Tensor> k,
    fine::ResourcePtr<Tensor> v,
    double scale,
    std::string mask_mode,
    std::vector<fine::ResourcePtr<Tensor>> mask_arrs,
    std::vector<fine::ResourcePtr<Tensor>> sinks_arrs) {
  return async_encoded(env, w,
      [q = std::move(q), k = std::move(k), v = std::move(v),
       scale, mask_mode = std::move(mask_mode),
       mask_arrs = std::move(mask_arrs),
       sinks_arrs = std::move(sinks_arrs)](mx::Stream &s) {
        std::optional<mx::array> mask_arr;
        if (!mask_arrs.empty()) {
          mask_arr = mask_arrs[0]->array;
        }
        std::optional<mx::array> sinks_arr;
        if (!sinks_arrs.empty()) {
          sinks_arr = sinks_arrs[0]->array;
        }
        return wrap(mx::fast::scaled_dot_product_attention(
            q->array, k->array, v->array, static_cast<float>(scale), mask_mode,
            mask_arr, sinks_arr, s));
      });
}
FINE_NIF(fast_scaled_dot_product_attention_nif, 0);

} // namespace
