// Convolutions.

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

fine::Term conv_general_nif(
    ErlNifEnv *env,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> input,
    fine::ResourcePtr<Tensor> weight,
    std::vector<int64_t> stride,
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> padding,
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> dilation,
    int64_t groups,
    bool flip) {
  return async_encoded(env, w,
      [input = std::move(input), weight = std::move(weight),
       stride = std::move(stride), padding = std::move(padding),
       dilation = std::move(dilation), groups, flip](mx::Stream &s) {
        return wrap(mx::conv_general(
            input->array, weight->array, to_int_vec(stride),
            to_int_vec(std::get<0>(padding)), to_int_vec(std::get<1>(padding)),
            to_int_vec(std::get<0>(dilation)), to_int_vec(std::get<1>(dilation)),
            static_cast<int>(groups), flip, s));
      });
}
FINE_NIF(conv_general_nif, 0);

} // namespace
