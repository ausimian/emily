// Miscellaneous ops: clip, roll, softmax, logcumsumexp, array_equal.

#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
using emily::Tensor;
using emily::WorkerThread;
using emily::to_int_vec;
using emily::wrap;

namespace {

fine::ResourcePtr<Tensor> clip(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> a_min,
    fine::ResourcePtr<Tensor> a_max) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::clip(a->array, a_min->array, a_max->array, s));
  });
}
FINE_NIF(clip, 0);

fine::ResourcePtr<Tensor> roll(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t shift,
    int64_t axis) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::roll(a->array, static_cast<int>(shift),
                         static_cast<int>(axis), s));
  });
}
FINE_NIF(roll, 0);

fine::ResourcePtr<Tensor> softmax(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    bool precise) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::softmax(a->array, to_int_vec(axes), precise, s));
  });
}
FINE_NIF(softmax, 0);

fine::ResourcePtr<Tensor> logcumsumexp(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool reverse,
    bool inclusive) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::logcumsumexp(a->array, static_cast<int>(axis), reverse,
                                 inclusive, s));
  });
}
FINE_NIF(logcumsumexp, 0);

fine::ResourcePtr<Tensor> array_equal(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    bool equal_nan) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::array_equal(a->array, b->array, equal_nan, s));
  });
}
FINE_NIF(array_equal, 0);

} // namespace
