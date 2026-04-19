// Miscellaneous ops: clip, roll, softmax, logcumsumexp, array_equal.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <vector>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::to_int_vec;
using emily::wrap;
using emily::WorkerThread;

namespace {

fine::Term clip_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> a_min,
    fine::ResourcePtr<Tensor> a_max) {
  return async_encoded(env, w,
      [a = std::move(a), a_min = std::move(a_min),
       a_max = std::move(a_max)](mx::Stream &s) {
        return wrap(mx::clip(a->array, a_min->array, a_max->array, s));
      });
}
FINE_NIF(clip_nif, 0);

fine::Term roll_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t shift,
    int64_t axis) {
  return async_encoded(env, w, [a = std::move(a), shift, axis](mx::Stream &s) {
    return wrap(mx::roll(a->array, static_cast<int>(shift),
                         static_cast<int>(axis), s));
  });
}
FINE_NIF(roll_nif, 0);

fine::Term softmax_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    std::vector<int64_t> axes,
    bool precise) {
  return async_encoded(env, w,
      [a = std::move(a), axes = std::move(axes), precise](mx::Stream &s) {
        return wrap(mx::softmax(a->array, to_int_vec(axes), precise, s));
      });
}
FINE_NIF(softmax_nif, 0);

fine::Term logcumsumexp_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    int64_t axis,
    bool reverse,
    bool inclusive) {
  return async_encoded(env, w,
      [a = std::move(a), axis, reverse, inclusive](mx::Stream &s) {
        return wrap(mx::logcumsumexp(a->array, static_cast<int>(axis), reverse,
                                     inclusive, s));
      });
}
FINE_NIF(logcumsumexp_nif, 0);

fine::Term array_equal_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> a,
    fine::ResourcePtr<Tensor> b,
    bool equal_nan) {
  return async_encoded(env, w,
      [a = std::move(a), b = std::move(b), equal_nan](mx::Stream &s) {
        return wrap(mx::array_equal(a->array, b->array, equal_nan, s));
      });
}
FINE_NIF(array_equal_nif, 0);

} // namespace
