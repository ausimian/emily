// Binary elementwise: arithmetic, compare, logical, bitwise.

#include "../emily/async.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

namespace mx = mlx::core;
using emily::async_encoded;
using emily::Tensor;
using emily::wrap;
using emily::WorkerThread;

namespace {

#define EMILY_BINARY(op_name, mlx_fn)                                          \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> a,                                             \
      fine::ResourcePtr<Tensor> b) {                                           \
    return async_encoded(env, w,                                               \
        [a = std::move(a), b = std::move(b)](mx::Stream &s) {                  \
          return wrap(mlx_fn(a->array, b->array, s));                          \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

// Arithmetic
EMILY_BINARY(add,           mx::add)
EMILY_BINARY(subtract,      mx::subtract)
EMILY_BINARY(multiply,      mx::multiply)
EMILY_BINARY(divide,        mx::divide)
EMILY_BINARY(floor_divide,  mx::floor_divide)
EMILY_BINARY(remainder,     mx::remainder)
EMILY_BINARY(power,         mx::power)
EMILY_BINARY(maximum,       mx::maximum)
EMILY_BINARY(minimum,       mx::minimum)
EMILY_BINARY(logaddexp,     mx::logaddexp)
EMILY_BINARY(arctan2,       mx::arctan2)

// Compare
EMILY_BINARY(equal,         mx::equal)
EMILY_BINARY(not_equal,     mx::not_equal)
EMILY_BINARY(less,          mx::less)
EMILY_BINARY(less_equal,    mx::less_equal)
EMILY_BINARY(greater,       mx::greater)
EMILY_BINARY(greater_equal, mx::greater_equal)

// Logical
EMILY_BINARY(logical_and,   mx::logical_and)
EMILY_BINARY(logical_or,    mx::logical_or)

// Bitwise
EMILY_BINARY(bitwise_and,   mx::bitwise_and)
EMILY_BINARY(bitwise_or,    mx::bitwise_or)
EMILY_BINARY(bitwise_xor,   mx::bitwise_xor)
EMILY_BINARY(left_shift,    mx::left_shift)
EMILY_BINARY(right_shift,   mx::right_shift)

#undef EMILY_BINARY

} // namespace
