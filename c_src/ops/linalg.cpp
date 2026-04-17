// Linear algebra: matmul, tensordot, outer, inner, and affine int4/int8
// quantization primitives (quantize / dequantize / quantized_matmul).

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_int_vec;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> matmul(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::matmul(a->array, b->array, s));
  });
}
FINE_NIF(matmul, 0);

fine::ResourcePtr<Tensor> tensordot(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    std::vector<int64_t> axes_a,
    std::vector<int64_t> axes_b) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::tensordot(a->array, b->array, to_int_vec(axes_a),
                              to_int_vec(axes_b), s));
  });
}
FINE_NIF(tensordot, 0);

fine::ResourcePtr<Tensor> outer(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::outer(a->array, b->array, s));
  });
}
FINE_NIF(outer, 0);

fine::ResourcePtr<Tensor> inner(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::inner(a->array, b->array, s));
  });
}
FINE_NIF(inner, 0);

std::tuple<fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>>
quantize(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> wt,
    int64_t group_size,
    int64_t bits) {
  return w->run_sync([&](mx::Stream &s) {
    auto result = mx::quantize(wt->array, static_cast<int>(group_size),
                               static_cast<int>(bits), "affine",
                               std::nullopt, s);
    return std::make_tuple(wrap(std::move(result[0])),
                           wrap(std::move(result[1])),
                           wrap(std::move(result[2])));
  });
}
FINE_NIF(quantize, 0);

fine::ResourcePtr<Tensor> dequantize(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> w_q,
    fine::ResourcePtr<Tensor> scales,
    fine::ResourcePtr<Tensor> biases,
    int64_t group_size,
    int64_t bits) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::dequantize(w_q->array, scales->array, biases->array,
                               static_cast<int>(group_size),
                               static_cast<int>(bits), "affine",
                               std::nullopt, std::nullopt, s));
  });
}
FINE_NIF(dequantize, 0);

fine::ResourcePtr<Tensor> quantized_matmul(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> x,
    fine::ResourcePtr<Tensor> w_q,
    fine::ResourcePtr<Tensor> scales,
    fine::ResourcePtr<Tensor> biases,
    bool transpose,
    int64_t group_size,
    int64_t bits) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::quantized_matmul(x->array, w_q->array, scales->array,
                                     biases->array, transpose,
                                     static_cast<int>(group_size),
                                     static_cast<int>(bits), "affine", s));
  });
}
FINE_NIF(quantized_matmul, 0);

} // namespace
