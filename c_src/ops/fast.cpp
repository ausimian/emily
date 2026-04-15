// Fused transformer kernels from mlx::core::fast.
//
// These handwritten kernels beat the defn-composed equivalents on the
// transformer hot paths: RMSNorm (one kernel vs rsqrt+mean+multiply
// chain), LayerNorm (same for Welford+affine), RoPE (fused trig +
// interleave), and Scaled-Dot-Product Attention (QK^T → scale → mask →
// softmax → V as one dispatch instead of ~5). Elixir-side they're
// surfaced as `Emily.Fast.*` helpers callable from inside `defn`.
//
// Nullable inputs (weight/bias, the RoPE `base` override, the precomp
// `freqs`, per-tensor `offset`) marshal via `std::optional` — the same
// pattern `ops/random.cpp` uses for PRNG keys.

#include "../emily/tensor.hpp"

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
using emily::unwrap_all;
using emily::wrap;

namespace {

// -----------------------------------------------------------------
// Nullable tensor helper
// -----------------------------------------------------------------

std::optional<mx::array> opt_array(
    const std::optional<fine::ResourcePtr<Tensor>> &opt) {
  if (opt) return (*opt)->array;
  return std::nullopt;
}

// -----------------------------------------------------------------
// fast_rms_norm/3 — mx::fast::rms_norm(x, weight?, eps)
// -----------------------------------------------------------------
//
// Normalises the last axis of `x` by `rsqrt(mean(x^2) + eps)` and
// optionally multiplies by `weight` (vector of size last-axis). The
// weight is nil-able because some models (e.g. `pre_norm=False`
// variants) use unit-scale RMSNorm.
fine::ResourcePtr<Tensor> fast_rms_norm(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> x,
    std::optional<fine::ResourcePtr<Tensor>> weight,
    double eps) {
  return wrap(mx::fast::rms_norm(
      x->array, opt_array(weight), static_cast<float>(eps)));
}
FINE_NIF(fast_rms_norm, 0);

// -----------------------------------------------------------------
// fast_layer_norm/4 — mx::fast::layer_norm(x, weight?, bias?, eps)
// -----------------------------------------------------------------
//
// Welford-style LayerNorm over the last axis with optional affine
// (weight + bias). `weight` and `bias` are independently nullable to
// match MLX — e.g. `elementwise_affine=False` PyTorch modules map to
// both-nil.
fine::ResourcePtr<Tensor> fast_layer_norm(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> x,
    std::optional<fine::ResourcePtr<Tensor>> weight,
    std::optional<fine::ResourcePtr<Tensor>> bias,
    double eps) {
  return wrap(mx::fast::layer_norm(
      x->array,
      opt_array(weight),
      opt_array(bias),
      static_cast<float>(eps)));
}
FINE_NIF(fast_layer_norm, 0);

// -----------------------------------------------------------------
// fast_rope/7 — mx::fast::rope(x, dims, traditional, base?, scale, offset, freqs?)
// -----------------------------------------------------------------
//
// Fused rotary positional embedding. `dims` is the count of trailing
// dimensions that carry rotated components (typically head_dim; the
// remaining trailing dims, if any, are passed through). `traditional`
// selects the paired-interleave layout (`true`, per Meta / MLX) vs
// the split-half layout (`false`, per HuggingFace). `base` is the
// theta override (nil → use the `freqs` argument instead); `freqs` is
// a pre-computed 1-D tensor of inverse frequencies to support the
// Llama-3 / LongRoPE / linear / dynamic scaling strategies that
// Bumblebee implements outside of MLX.
//
// `offset` is a scalar integer tensor (Nx's canonical rep — Bumblebee
// tracks the cumulative position offset as an %Nx.Tensor{} through
// iterative decode), which matches the `array`-offset overload of
// MLX's `rope`. The NIF always takes a tensor here and uses the
// overload with `const array&` — users pass `Nx.tensor(0)` when
// there's no KV-cache offset.
fine::ResourcePtr<Tensor> fast_rope(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> x,
    int64_t dims,
    bool traditional,
    std::optional<double> base,
    double scale,
    fine::ResourcePtr<Tensor> offset,
    std::optional<fine::ResourcePtr<Tensor>> freqs) {
  std::optional<float> base_f;
  if (base) base_f = static_cast<float>(*base);

  return wrap(mx::fast::rope(
      x->array,
      static_cast<int>(dims),
      traditional,
      base_f,
      static_cast<float>(scale),
      offset->array,
      opt_array(freqs)));
}
FINE_NIF(fast_rope, 0);

// -----------------------------------------------------------------
// fast_scaled_dot_product_attention/6 —
//   mx::fast::scaled_dot_product_attention(Q, K, V, scale, mask_mode, mask_arrs)
// -----------------------------------------------------------------
//
// Computes `softmax((Q @ Kᵀ) * scale + mask) @ V` as a single fused
// kernel over `[B, H, S, D]` inputs.
//
// `mask_mode` is the empty string, `"causal"`, or `"array"`:
//   - `""`        — no mask.
//   - `"causal"`  — upper-triangular -inf mask (no additional arrays).
//   - `"array"`   — `mask_arrs` holds one broadcastable additive bias
//                   tensor (Bumblebee's `bias = select(mask, 0, -inf)`
//                   materialises this).
//
// MLX supports a handful of other modes (block-sparse etc.) — out of
// scope for M11; add them when a model asks.
fine::ResourcePtr<Tensor> fast_scaled_dot_product_attention(
    ErlNifEnv *,
    fine::ResourcePtr<Tensor> q,
    fine::ResourcePtr<Tensor> k,
    fine::ResourcePtr<Tensor> v,
    double scale,
    std::string mask_mode,
    std::vector<fine::ResourcePtr<Tensor>> mask_arrs) {
  return wrap(mx::fast::scaled_dot_product_attention(
      q->array,
      k->array,
      v->array,
      static_cast<float>(scale),
      mask_mode,
      unwrap_all(mask_arrs)));
}
FINE_NIF(fast_scaled_dot_product_attention, 0);

} // namespace
