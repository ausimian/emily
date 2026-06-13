// Pure op cores: the `mx::array`-building expression for each Emily op,
// free of NIF / async / enif plumbing. One source of truth per op,
// called from BOTH the eager per-op NIF (c_src/ops/*.cpp) and the
// Expr-compiler program replay (c_src/program.cpp). Keeping the core
// here means the compiled (single-NIF) path can never numerically drift
// from the eager path — they invoke the same function.
//
// CM0 ships only `add` (the prototype op). CM1 fills in the full
// primitive set as each c_src/ops/*.cpp op is split into core + thin NIF.

#pragma once

#include <mlx/mlx.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace emily::ops {

namespace mx = mlx::core;

// --- Binary elementwise ---

inline mx::array add_core(const mx::array &a, const mx::array &b,
                          mx::Stream &s) {
  return mx::add(a, b, s);
}

// --- Shape ---

// Reverse `a` along `axis` via a negative-stride slice. `axis` may be
// negative (normalized against the rank); a scalar (ndim 0) is returned
// unchanged. The slice bounds are built as mx::Shape directly because the
// stop sentinel (`-dim-1`) is negative, which the shape-validation helper
// (to_mlx_shape) rejects by design.
inline mx::array flip_core(const mx::array &a, int64_t axis, mx::Stream &s) {
  const auto &shape = a.shape();
  const int ndim = static_cast<int>(shape.size());
  if (ndim == 0) {
    return a;
  }
  int ax = static_cast<int>(axis);
  if (ax < 0) {
    ax += ndim;
  }
  if (ax < 0 || ax >= ndim) {
    throw std::invalid_argument("[flip] axis " + std::to_string(axis) +
                                " out of range for ndim " +
                                std::to_string(ndim));
  }
  mx::Shape starts(ndim, 0);
  mx::Shape stops(shape.begin(), shape.end());
  mx::Shape strides(ndim, 1);
  starts[ax] = shape[ax] - 1;
  stops[ax] = -shape[ax] - 1;
  strides[ax] = -1;
  return mx::slice(a, std::move(starts), std::move(stops), std::move(strides),
                   s);
}

// --- Window / pooling (forward reductions) ---
//
// MLX exposes no window_sum/max/min/product primitive; each is composed
// as pad -> as_strided (sliding-window view) -> reduce over the kernel
// axes. These cores back both the eager NIFs (c_src/ops/pooling.cpp) and
// the compiled program replay, so the two paths can't drift.

// Contiguous element-strides for a shape, e.g. {B, H, W, C} ->
// {H*W*C, W*C, C, 1}.
inline mx::Strides contiguous_strides(const mx::Shape &shape) {
  int rank = static_cast<int>(shape.size());
  mx::Strides out(rank, 1);
  for (int i = rank - 2; i >= 0; --i) {
    out[i] = out[i + 1] * static_cast<int64_t>(shape[i + 1]);
  }
  return out;
}

// Pad `a` with `pad_value` using per-axis lo/hi pads. Returns `a`
// unchanged if all pads are zero (the common path — avoids a copy).
inline mx::array do_pad(
    const mx::array &a,
    const std::vector<int64_t> &pad_lo,
    const std::vector<int64_t> &pad_hi,
    const mx::array &pad_value,
    mx::Stream &s) {
  int rank = static_cast<int>(a.ndim());
  if (pad_lo.size() != static_cast<std::size_t>(rank) ||
      pad_hi.size() != static_cast<std::size_t>(rank)) {
    throw std::invalid_argument(
        "pad: pad_lo/pad_hi length must equal tensor rank " +
        std::to_string(rank));
  }
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
// `out_dims` is filled with the per-axis output size.
//
//   eff_window = (window_shape[i] - 1) * dilations[i] + 1
//   out[i]     = (padded_shape[i] - eff_window) / strides[i] + 1
inline mx::array sliding_windows_view(
    const mx::array &padded,
    const std::vector<int64_t> &window_shape,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &dilations,
    std::vector<int64_t> &out_dims,
    mx::Stream &s) {
  int rank = static_cast<int>(padded.ndim());
  const auto rank_sz = static_cast<std::size_t>(rank);
  if (window_shape.size() != rank_sz || strides.size() != rank_sz ||
      dilations.size() != rank_sz) {
    throw std::invalid_argument(
        "window: window_shape/strides/dilations length must equal tensor "
        "rank " +
        std::to_string(rank));
  }
  for (int i = 0; i < rank; ++i) {
    if (window_shape[i] < 1 || strides[i] < 1 || dilations[i] < 1) {
      throw std::invalid_argument(
          "window: window dimensions, strides, and dilations must all be "
          "positive");
    }
  }
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

enum class WindowReduceKind { Sum, Max, Min, Product };

// pad -> sliding-window view -> reduce over the kernel axes. `init_value`
// is the dtype identity (0/1/±inf), used both as the pad fill and (for
// max/min) the reduction's boundary identity.
inline mx::array window_reduce_core(
    const mx::array &a,
    const std::vector<int64_t> &window_shape,
    const std::vector<int64_t> &strides,
    const std::vector<int64_t> &pad_lo,
    const std::vector<int64_t> &pad_hi,
    const std::vector<int64_t> &dilations,
    const mx::array &init_value,
    WindowReduceKind kind,
    mx::Stream &s) {
  auto padded = do_pad(a, pad_lo, pad_hi, init_value, s);
  std::vector<int64_t> out_dims;
  auto view =
      sliding_windows_view(padded, window_shape, strides, dilations, out_dims, s);

  // Dilated windows give the kernel axes an `as_strided` stride > 1, so the
  // view aliases fewer physical elements than its logical size (overlapping
  // strides). MLX's reduction then picks a strided fast path
  // (GeneralStridedReduce) that assumes a dense, non-overlapping layout and
  // reads `product(shape)` contiguous elements — over-running the buffer and
  // returning garbage for windows past the first stride positions (issue
  // #175). Materialise the view first: the general copy reads element-by-
  // element via the real strides (always in-bounds, since the last window's
  // last tap is the last real element), yielding a dense buffer the reducer
  // can safely fast-path. Only dilated windows need this; the common
  // (non-dilated) pooling path keeps its copy-free strided reduce.
  bool dilated = false;
  for (int64_t d : dilations) {
    if (d > 1) {
      dilated = true;
      break;
    }
  }
  if (dilated) {
    view = mx::contiguous(view, /*allow_col_major=*/false, s);
  }

  int rank = static_cast<int>(window_shape.size());
  std::vector<int> reduce_axes(rank);
  for (int i = 0; i < rank; ++i)
    reduce_axes[i] = rank + i;

  switch (kind) {
  case WindowReduceKind::Sum:
    return mx::sum(view, reduce_axes, /*keepdims=*/false, s);
  case WindowReduceKind::Max:
    return mx::max(view, reduce_axes, /*keepdims=*/false, s);
  case WindowReduceKind::Min:
    return mx::min(view, reduce_axes, /*keepdims=*/false, s);
  case WindowReduceKind::Product:
    return mx::prod(view, reduce_axes, /*keepdims=*/false, s);
  }
  throw std::invalid_argument("window_reduce_core: unknown reduce kind");
}

// Select-and-scatter — the backward of window_max/window_min (Nx rewrites
// grad(window_max) into window_scatter_max). `is_max` picks argmax vs
// argmin. Tie-break: Nx's select_and_scatter uses `>=`/`<=` (LAST
// occurrence); MLX argmax/argmin give FIRST, so we argmax `mask * pos`
// to recover the last winner. Scatter variants take no dilations.
inline mx::array window_scatter_core(
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

  // 2. Sliding-window view (dilation is implicitly 1 per axis for scatter).
  std::vector<int64_t> dilations(rank, 1);
  std::vector<int64_t> out_dims;
  auto view =
      sliding_windows_view(padded, window_shape, strides, dilations, out_dims, s);

  // 3. Flatten the kernel axes so a single reduction spans the window.
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

  // 4. Argmax-with-tie-break (mask*pos picks the last-occurrence winner).
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

  // 5. Decompose the flat kernel index into per-axis kernel indices.
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

  // 6. Per-axis absolute indices into the padded tensor.
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

  // 7. Reshape source so each index tuple is a single-point write.
  mx::Shape source_reshape;
  source_reshape.reserve(2 * rank);
  for (int i = 0; i < rank; ++i)
    source_reshape.push_back(static_cast<mx::ShapeElem>(out_dims[i]));
  for (int i = 0; i < rank; ++i)
    source_reshape.push_back(1);
  auto source_r = mx::reshape(source, source_reshape, s);
  source_r = mx::astype(source_r, tensor.dtype(), s);

  // 8. Output buffer starts filled with init_value (unselected positions
  //    retain it; selected positions receive init_value + sum(source)).
  auto padded_out = mx::full(padded_shape, init_value, tensor.dtype(), s);

  // 9. Scatter-add all selected contributions in one dispatch.
  std::vector<int> axes(rank);
  std::iota(axes.begin(), axes.end(), 0);
  auto scattered = mx::scatter_add(padded_out, abs_indices, source_r, axes, s);

  // 10. Slice back to the original (unpadded) shape.
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

} // namespace emily::ops
