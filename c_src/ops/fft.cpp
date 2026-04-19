// Fast Fourier Transforms.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
namespace fft = mlx::core::fft;
using emily::async_encoded;
using emily::Tensor;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term fftn_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), n = std::move(n),
       axes = std::move(axes)](mx::Stream &s) {
        return wrap(fft::fftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                              fft::FFTNorm::Backward, s));
      });
}
FINE_NIF(fftn_nif, 0);

fine::Term ifftn_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), n = std::move(n),
       axes = std::move(axes)](mx::Stream &s) {
        return wrap(fft::ifftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                               fft::FFTNorm::Backward, s));
      });
}
FINE_NIF(ifftn_nif, 0);

fine::Term rfftn_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), n = std::move(n),
       axes = std::move(axes)](mx::Stream &s) {
        return wrap(fft::rfftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                               fft::FFTNorm::Backward, s));
      });
}
FINE_NIF(rfftn_nif, 0);

fine::Term irfftn_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return async_encoded(env, w,
      [a = std::move(a), n = std::move(n),
       axes = std::move(axes)](mx::Stream &s) {
        return wrap(fft::irfftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                                fft::FFTNorm::Backward, s));
      });
}
FINE_NIF(irfftn_nif, 0);

} // namespace
