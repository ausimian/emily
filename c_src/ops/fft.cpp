// Fast Fourier Transforms.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
namespace fft = mlx::core::fft;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_int_vec;
using emily::to_mlx_shape;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> fftn(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(fft::fftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                          fft::FFTNorm::Backward, s));
  });
}
FINE_NIF(fftn, 0);

fine::ResourcePtr<Tensor> ifftn(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(fft::ifftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                           fft::FFTNorm::Backward, s));
  });
}
FINE_NIF(ifftn, 0);

fine::ResourcePtr<Tensor> rfftn(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(fft::rfftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                           fft::FFTNorm::Backward, s));
  });
}
FINE_NIF(rfftn, 0);

fine::ResourcePtr<Tensor> irfftn(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> n,
    std::vector<int64_t> axes) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(fft::irfftn(a->array, to_mlx_shape(n), to_int_vec(axes),
                            fft::FFTNorm::Backward, s));
  });
}
FINE_NIF(irfftn, 0);

} // namespace
