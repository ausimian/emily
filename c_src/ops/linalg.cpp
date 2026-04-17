// Linear algebra: matmul, tensordot, outer, inner, decompositions
// (lu, svd, qr, cholesky, eigh), solvers (solve, solve_triangular),
// and affine int4/int8 quantization primitives
// (quantize / dequantize / quantized_matmul).

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

// ---- Decompositions / solvers (mx::linalg::*) ------------------
//
// MLX's linalg primitives are CPU-only — they throw on a GPU stream.
// Each NIF dispatches via the worker's run_sync (serialisation) but
// uses the CPU default stream for the actual linalg call. MLX handles
// cross-stream data dependencies internally via its lazy eval graph.

// LU decomposition. Returns {P, L, U}.
std::tuple<fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>>
linalg_lu(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto result = mx::linalg::lu(a->array, cpu);
    return std::make_tuple(
        wrap(std::move(result[0])),
        wrap(std::move(result[1])),
        wrap(std::move(result[2])));
  });
}
FINE_NIF(linalg_lu, 0);

// Singular value decomposition. Returns {U, S, Vt}.
std::tuple<fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>>
linalg_svd(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto result = mx::linalg::svd(a->array, true, cpu);
    return std::make_tuple(
        wrap(std::move(result[0])),
        wrap(std::move(result[1])),
        wrap(std::move(result[2])));
  });
}
FINE_NIF(linalg_svd, 0);

// QR decomposition (reduced). Returns {Q, R}.
std::tuple<fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>>
linalg_qr(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto [q, r] = mx::linalg::qr(a->array, cpu);
    return std::make_tuple(wrap(std::move(q)), wrap(std::move(r)));
  });
}
FINE_NIF(linalg_qr, 0);

// Cholesky decomposition. `upper` selects upper- vs lower-triangular.
fine::ResourcePtr<Tensor> linalg_cholesky(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    bool upper) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    return wrap(mx::linalg::cholesky(a->array, upper, cpu));
  });
}
FINE_NIF(linalg_cholesky, 0);

// Symmetric eigendecomposition. Returns {eigenvalues, eigenvectors}.
std::tuple<fine::ResourcePtr<Tensor>,
           fine::ResourcePtr<Tensor>>
linalg_eigh(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::string uplo) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    auto [vals, vecs] = mx::linalg::eigh(a->array, uplo, cpu);
    return std::make_tuple(wrap(std::move(vals)), wrap(std::move(vecs)));
  });
}
FINE_NIF(linalg_eigh, 0);

// General linear solve: A X = B.
fine::ResourcePtr<Tensor> linalg_solve(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    return wrap(mx::linalg::solve(a->array, b->array, cpu));
  });
}
FINE_NIF(linalg_solve, 0);

// Triangular solve: A X = B where A is upper- or lower-triangular.
fine::ResourcePtr<Tensor> linalg_solve_triangular(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    bool upper) {
  return w->run_sync([&](mx::Stream & /*s*/) {
    auto cpu = mx::default_stream(mx::Device(mx::Device::DeviceType::cpu));
    return wrap(mx::linalg::solve_triangular(a->array, b->array, upper, cpu));
  });
}
FINE_NIF(linalg_solve_triangular, 0);

} // namespace
