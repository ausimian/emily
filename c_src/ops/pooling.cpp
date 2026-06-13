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
#include "../emily/op_cores.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::wrap;
using emily::WorkerThread;

namespace {

// The window cores (do_pad / sliding_windows_view / window_reduce_core)
// live in emily/op_cores.hpp so the eager NIFs below and the Expr-compiler
// program replay (c_src/program.cpp) share one implementation and can't
// numerically drift. window_scatter_impl below still composes do_pad +
// sliding_windows_view from there.
using emily::ops::do_pad;
using emily::ops::sliding_windows_view;
using emily::ops::window_reduce_core;
using emily::ops::window_scatter_core;
using emily::ops::WindowReduceKind;

// -------------------- Reductions --------------------

#define EMILY_WINDOW_REDUCE(op_name, kind)                                    \
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
          return wrap(window_reduce_core(                                      \
              t->array, window_shape, strides, pad_lo, pad_hi, dilations,      \
              init_value->array, kind, s));                                    \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_WINDOW_REDUCE(window_sum,     WindowReduceKind::Sum)
EMILY_WINDOW_REDUCE(window_max,     WindowReduceKind::Max)
EMILY_WINDOW_REDUCE(window_min,     WindowReduceKind::Min)
EMILY_WINDOW_REDUCE(window_product, WindowReduceKind::Product)

#undef EMILY_WINDOW_REDUCE

// -------------------- Scatter variants --------------------
//
// window_scatter_core (the MaxPool/MinPool backward) lives in
// emily/op_cores.hpp so the eager NIFs below and the Expr-compiler program
// replay share one implementation.

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
        return wrap(window_scatter_core(
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
        return wrap(window_scatter_core(
            t->array, source->array, init_value->array, window_shape, strides,
            pad_lo, pad_hi, /*is_max=*/false, s));
      });
}
FINE_NIF(window_scatter_min_nif, 0);

} // namespace
