// Window / pooling reductions and scatters.
//
// MLX exposes no direct window_sum/max/min/product primitives. We
// compose each op as pad -> as_strided (sliding-window view) -> reduce,
// mirroring MLX's own nn/layers/pooling.py but generalised to N-D (Nx
// passes a per-axis window shape, strides, padding, and dilation).
//
// window_scatter_max/min are on the MaxPool backward path (Nx rewrites
// grad(window_max) into window_scatter_max), so we lift them here too
// rather than leave the backward pass on via_binary. The scatter path:
// pad -> as_strided -> flatten kernel axes -> argmax-with-tie-break
// (last-occurrence per Nx semantics) -> decompose flat arg into
// per-axis kernel indices -> build absolute padded-coord indices ->
// scatter_add source values into a full(init_value) tensor -> slice
// back to the unpadded shape.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <numeric>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::wrap;
using emily::WorkerThread;

namespace {

// -------------------- Shared helpers --------------------

// Contiguous element-strides for a shape, e.g. {B, H, W, C} ->
// {H*W*C, W*C, C, 1}.
mx::Strides contiguous_strides(const mx::Shape &shape) {
  int rank = static_cast<int>(shape.size());
  mx::Strides out(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    out[i] = out[i + 1] * static_cast<int64_t>(shape[i + 1]);
  }
  return out;
}

// Pad `a` with `pad_value` using per-axis lo/hi pads. Returns `a`
// unchanged if all pads are zero (the common path — avoids a pointless
// copy).
mx::array do_pad(
    const mx::array &a,
    const std::vector<int64_t> &pad_lo,
    const std::vector<int64_t> &pad_hi,
    const mx::array &pad_value,
    mx::Stream &s) {
  int rank = static_cast<int>(a.ndim());
  bool any_pad = false;
  for (int i = 0; i < rank; ++i) {
    if (pad_lo[i] > 0 || pad_hi[i] > 0) {
      any_pad = true;
      break;
    }
  }
  if (!any_pad) {
    return a;
  }

  std::vector<int> axes(rank);
  std::iota(axes.begin(), axes.end(), 0);

  mx::Shape lo, hi;
  lo.reserve(rank);
  hi.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    lo.push_back(static_cast<mx::ShapeElem>(pad_lo[i]));
    hi.push_back(static_cast<mx::ShapeElem>(pad_hi[i]));
  }

  return mx::pad(a, axes, lo, hi, pad_value, "constant", s);
}

// Build an `as_strided` view with shape `[out_dims..., window_shape...]`.
// Output `out_dims` is filled with the per-axis output size.
//
// Output shape formula (per axis):
//   eff_window = (window_shape[i] - 1) * dilations[i] + 1
//   out[i]     = (padded_shape[i] - eff_window) / strides[i] + 1
//
// Strides (in elements, relative to the padded tensor's contiguous
// layout — `as_strided` forces its input to be contiguous internally):
//   out-axis  i: contiguous_stride[i] * strides[i]
//   kernel-ax i: contiguous_stride[i] * dilations[i]
mx::array sliding_windows_view(
    const mx::array &padded,
    const std::vector<int64_t> &window_shape,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &dilations,
    std::vector<int64_t> &out_dims,
    mx::Stream &s) {
  int rank = static_cast<int>(padded.ndim());
  const auto &padded_shape = padded.shape();
  auto cs = contiguous_strides(padded_shape);

  out_dims.assign(rank, 0);
  mx::Shape new_shape;
  mx::Strides new_strides;
  new_shape.reserve(2 * rank);
  new_strides.reserve(2 * rank);

  for (int i = 0; i < rank; ++i) {
    int64_t eff = (window_shape[i] - 1) * dilations[i] + 1;
    out_dims[i] = (static_cast<int64_t>(padded_shape[i]) - eff) / strides[i] + 1;
    new_shape.push_back(static_cast<mx::ShapeElem>(out_dims[i]));
  }
  for (int i = 0; i < rank; ++i) {
    new_shape.push_back(static_cast<mx::ShapeElem>(window_shape[i]));
  }
  for (int i = 0; i < rank; ++i) {
    new_strides.push_back(cs[i] * strides[i]);
  }
  for (int i = 0; i < rank; ++i) {
    new_strides.push_back(cs[i] * dilations[i]);
  }

  return mx::as_strided(padded, new_shape, new_strides, 0, s);
}

// -------------------- Reductions --------------------

#define EMILY_WINDOW_REDUCE(op_name, mlx_fn)                                   \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> t,                                             \
      std::vector<int64_t> window_shape,                                       \
      std::vector<int64_t> strides,                                            \
      std::vector<int64_t> pad_lo,                                             \
      std::vector<int64_t> pad_hi,                                             \
      std::vector<int64_t> dilations,                                          \
      fine::ResourcePtr<Tensor> init_value) {                                  \
    return async_encoded(env, w,                                               \
        [t = std::move(t), window_shape = std::move(window_shape),             \
         strides = std::move(strides), pad_lo = std::move(pad_lo),             \
         pad_hi = std::move(pad_hi), dilations = std::move(dilations),         \
         init_value = std::move(init_value)](mx::Stream &s) {                  \
          auto padded = do_pad(t->array, pad_lo, pad_hi, init_value->array, s);\
          std::vector<int64_t> out_dims;                                       \
          auto view = sliding_windows_view(padded, window_shape, strides,      \
                                           dilations, out_dims, s);            \
          int rank = static_cast<int>(window_shape.size());                    \
          std::vector<int> reduce_axes(rank);                                  \
          for (int i = 0; i < rank; ++i)                                       \
            reduce_axes[i] = rank + i;                                         \
          return wrap(mlx_fn(view, reduce_axes, /*keepdims=*/false, s));       \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_WINDOW_REDUCE(window_sum,     mx::sum)
EMILY_WINDOW_REDUCE(window_max,     mx::max)
EMILY_WINDOW_REDUCE(window_min,     mx::min)
EMILY_WINDOW_REDUCE(window_product, mx::prod)

#undef EMILY_WINDOW_REDUCE

// -------------------- Scatter variants --------------------

// Shared body: select-and-scatter. `is_max` picks between argmax
// (for window_scatter_max) and argmin (for window_scatter_min).
//
// Tie-break semantics: Nx uses `>=` / `<=` in its select_and_scatter,
// i.e. the LAST-occurrence winner. MLX's argmax/argmin return
// FIRST-occurrence. We build `mask_pos = (flat_view == selector) *
// arange(K)` and argmax that — for tied positions the pos multiplier
// makes the later index strictly larger, giving last-occurrence.
mx::array window_scatter_impl(
    const mx::array &tensor,
    const mx::array &source,
    const mx::array &init_value,
    const std::vector<int64_t> &window_shape,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &pad_lo,
    const std::vector<int64_t> &pad_hi,
    bool is_max,
    mx::Stream &s) {
  int rank = static_cast<int>(window_shape.size());
  auto original_shape = tensor.shape();

  // 1. Pad input with init_value.
  auto padded = do_pad(tensor, pad_lo, pad_hi, init_value, s);
  auto padded_shape = padded.shape();

  // 2. Sliding-window view. Scatter variants don't take dilations in
  //    Nx's API, so dilation is implicitly 1 per axis.
  std::vector<int64_t> dilations(rank, 1);
  std::vector<int64_t> out_dims;
  auto view =
      sliding_windows_view(padded, window_shape, strides, dilations, out_dims, s);

  // 3. Flatten the kernel axes so we can argmax across the whole window
  //    in a single reduction.
  int64_t K = 1;
  for (int i = 0; i < rank; ++i)
    K *= window_shape[i];

  mx::Shape flat_view_shape;
  flat_view_shape.reserve(rank + 1);
  for (int i = 0; i < rank; ++i)
    flat_view_shape.push_back(static_cast<mx::ShapeElem>(out_dims[i]));
  flat_view_shape.push_back(static_cast<mx::ShapeElem>(K));

  auto flat_view = mx::reshape(view, flat_view_shape, s);
  int last_axis = rank;

  // 4. Argmax-with-tie-break. `selector` is the per-window max/min; mask
  //    is 1 where the kernel element equals the winner; mask*pos gives
  //    later-matching positions a higher value, so argmax picks the
  //    last-occurrence kernel index.
  auto selector = is_max
                      ? mx::max(flat_view, last_axis, /*keepdims=*/true, s)
                      : mx::min(flat_view, last_axis, /*keepdims=*/true, s);
  auto mask = mx::equal(flat_view, selector, s);

  auto pos_1d = mx::arange(0.0, static_cast<double>(K), 1.0, mx::int32, s);
  mx::Shape pos_shape(rank + 1, 1);
  pos_shape[rank] = static_cast<mx::ShapeElem>(K);
  auto pos = mx::reshape(pos_1d, pos_shape, s);

  auto mask_i = mx::astype(mask, mx::int32, s);
  auto mask_pos = mx::multiply(mask_i, pos, s);
  auto last_arg = mx::argmax(mask_pos, last_axis, /*keepdims=*/false, s);

  // 5. Decompose flat kernel index into per-axis kernel indices.
  //    k_idx[R-1] = last_arg % window[R-1];
  //    k_idx[R-2] = (last_arg / window[R-1]) % window[R-2]; ...
  std::vector<mx::array> k_idx;
  k_idx.reserve(rank);
  for (int i = 0; i < rank; ++i)
    k_idx.push_back(last_arg); // placeholder; overwritten below

  mx::array remaining = last_arg;
  for (int i = rank - 1; i >= 0; --i) {
    auto w_i = mx::array(static_cast<int32_t>(window_shape[i]), mx::int32);
    k_idx[i] = mx::remainder(remaining, w_i, s);
    if (i > 0) {
      remaining = mx::floor_divide(remaining, w_i, s);
    }
  }

  // 6. Per-axis absolute indices into the padded tensor:
  //    abs_idx[i][out_coord] = out_coord[i] * stride[i] + k_idx[i][out_coord]
  mx::Shape out_shape_s;
  out_shape_s.reserve(rank);
  for (int i = 0; i < rank; ++i)
    out_shape_s.push_back(static_cast<mx::ShapeElem>(out_dims[i]));

  std::vector<mx::array> abs_indices;
  abs_indices.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    auto base_i =
        mx::arange(0.0, static_cast<double>(out_dims[i]), 1.0, mx::int32, s);
    mx::Shape bcast(rank, 1);
    bcast[i] = static_cast<mx::ShapeElem>(out_dims[i]);
    base_i = mx::reshape(base_i, bcast, s);
    auto stride_i = mx::array(static_cast<int32_t>(strides[i]), mx::int32);
    auto base_times = mx::multiply(base_i, stride_i, s);
    auto bt = mx::broadcast_to(base_times, out_shape_s, s);
    abs_indices.push_back(mx::add(bt, k_idx[i], s));
  }

  // 7. Reshape source to IDX_SHAPE + [1]*rank so MLX scatter_add treats
  //    each index tuple as a single-point write.
  mx::Shape source_reshape;
  source_reshape.reserve(2 * rank);
  for (int i = 0; i < rank; ++i)
    source_reshape.push_back(static_cast<mx::ShapeElem>(out_dims[i]));
  for (int i = 0; i < rank; ++i)
    source_reshape.push_back(1);
  auto source_r = mx::reshape(source, source_reshape, s);
  source_r = mx::astype(source_r, tensor.dtype(), s);

  // 8. Output buffer starts filled with init_value. Matches Nx's
  //    select_and_scatter: unselected positions retain init_value;
  //    selected positions receive init_value + sum(source values).
  auto padded_out = mx::full(padded_shape, init_value, tensor.dtype(), s);

  // 9. Scatter-add all selected contributions in one dispatch.
  std::vector<int> axes(rank);
  std::iota(axes.begin(), axes.end(), 0);
  auto scattered = mx::scatter_add(padded_out, abs_indices, source_r, axes, s);

  // 10. Slice back to original (unpadded) shape.
  mx::Shape slice_start, slice_stop, slice_strides_v;
  slice_start.reserve(rank);
  slice_stop.reserve(rank);
  slice_strides_v.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    slice_start.push_back(static_cast<mx::ShapeElem>(pad_lo[i]));
    slice_stop.push_back(
        static_cast<mx::ShapeElem>(pad_lo[i] + original_shape[i]));
    slice_strides_v.push_back(1);
  }
  return mx::slice(scattered, slice_start, slice_stop, slice_strides_v, s);
}

fine::Term window_scatter_max_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> t,
    fine::ResourcePtr<Tensor> source,
    fine::ResourcePtr<Tensor> init_value,
    std::vector<int64_t> window_shape,
    std::vector<int64_t> strides,
    std::vector<int64_t> pad_lo,
    std::vector<int64_t> pad_hi) {
  return async_encoded(env, w,
      [t = std::move(t), source = std::move(source),
       init_value = std::move(init_value),
       window_shape = std::move(window_shape),
       strides = std::move(strides), pad_lo = std::move(pad_lo),
       pad_hi = std::move(pad_hi)](mx::Stream &s) {
        return wrap(window_scatter_impl(
            t->array, source->array, init_value->array, window_shape, strides,
            pad_lo, pad_hi, /*is_max=*/true, s));
      });
}
FINE_NIF(window_scatter_max_nif, 0);

fine::Term window_scatter_min_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> t,
    fine::ResourcePtr<Tensor> source,
    fine::ResourcePtr<Tensor> init_value,
    std::vector<int64_t> window_shape,
    std::vector<int64_t> strides,
    std::vector<int64_t> pad_lo,
    std::vector<int64_t> pad_hi) {
  return async_encoded(env, w,
      [t = std::move(t), source = std::move(source),
       init_value = std::move(init_value),
       window_shape = std::move(window_shape),
       strides = std::move(strides), pad_lo = std::move(pad_lo),
       pad_hi = std::move(pad_hi)](mx::Stream &s) {
        return wrap(window_scatter_impl(
            t->array, source->array, init_value->array, window_shape, strides,
            pad_lo, pad_hi, /*is_max=*/false, s));
      });
}
FINE_NIF(window_scatter_min_nif, 0);

} // namespace
