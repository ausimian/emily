// Convolutions.

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

fine::ResourcePtr<Tensor> conv_general(
    ErlNifEnv *,
    fine::ResourcePtr<WorkerThread> w,
    fine::ResourcePtr<Tensor> input,
    fine::ResourcePtr<Tensor> weight,
    std::vector<int64_t> stride,
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> padding,
    std::tuple<std::vector<int64_t>, std::vector<int64_t>> dilation,
    int64_t groups,
    bool flip) {
  return w->run_sync([&](mx::Stream &s) {
    return wrap(mx::conv_general(
        input->array, weight->array, to_int_vec(stride),
        to_int_vec(std::get<0>(padding)), to_int_vec(std::get<1>(padding)),
        to_int_vec(std::get<0>(dilation)), to_int_vec(std::get<1>(dilation)),
        static_cast<int>(groups), flip, s));
  });
}
FINE_NIF(conv_general, 0);

} // namespace
