// Unary elementwise ops. All take a Tensor and return a Tensor.

#include "../emily/tensor.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

namespace mx = mlx::core;
using emily::Tensor;
using emily::wrap;

// Anonymous namespace so our NIF names (log1p, sqrt, sin, etc.) don't
// clash with C math-library functions brought in by MLX headers.
namespace {

#define EMILY_UNARY(nif_name, mlx_fn)                                          \
  fine::ResourcePtr<Tensor> nif_name(                                          \
      ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t s) {                   \
    return wrap(mlx_fn(a->array, emily::resolve_stream(s)));                   \
  }                                                                            \
  FINE_NIF(nif_name, 0);

EMILY_UNARY(negative,        mx::negative)
EMILY_UNARY(abs,             mx::abs)
EMILY_UNARY(sign,            mx::sign)
EMILY_UNARY(floor,           mx::floor)
EMILY_UNARY(ceil,            mx::ceil)
EMILY_UNARY(sqrt,            mx::sqrt)
EMILY_UNARY(rsqrt,           mx::rsqrt)
EMILY_UNARY(exp,             mx::exp)
EMILY_UNARY(expm1,           mx::expm1)
EMILY_UNARY(log,             mx::log)
EMILY_UNARY(log1p,           mx::log1p)
EMILY_UNARY(log2,            mx::log2)
EMILY_UNARY(log10,           mx::log10)
EMILY_UNARY(sin,             mx::sin)
EMILY_UNARY(cos,             mx::cos)
EMILY_UNARY(tan,             mx::tan)
EMILY_UNARY(arcsin,          mx::arcsin)
EMILY_UNARY(arccos,          mx::arccos)
EMILY_UNARY(arctan,          mx::arctan)
EMILY_UNARY(sinh,            mx::sinh)
EMILY_UNARY(cosh,            mx::cosh)
EMILY_UNARY(tanh,            mx::tanh)
EMILY_UNARY(arcsinh,         mx::arcsinh)
EMILY_UNARY(arccosh,         mx::arccosh)
EMILY_UNARY(arctanh,         mx::arctanh)
EMILY_UNARY(sigmoid,         mx::sigmoid)
EMILY_UNARY(erf,             mx::erf)
EMILY_UNARY(erfinv,          mx::erfinv)
EMILY_UNARY(square,          mx::square)
EMILY_UNARY(reciprocal,      mx::reciprocal)
EMILY_UNARY(logical_not,     mx::logical_not)
EMILY_UNARY(bitwise_invert,  mx::bitwise_invert)
EMILY_UNARY(isnan,           mx::isnan)
EMILY_UNARY(isinf,           mx::isinf)
EMILY_UNARY(isfinite,        mx::isfinite)
EMILY_UNARY(conjugate,       mx::conjugate)
EMILY_UNARY(real,            mx::real)
EMILY_UNARY(imag,            mx::imag)
EMILY_UNARY(stop_gradient,   mx::stop_gradient)

#undef EMILY_UNARY

// round/2 takes an extra decimals arg; handled separately.
fine::ResourcePtr<Tensor> round(
    ErlNifEnv *, fine::ResourcePtr<Tensor> a, int64_t decimals, int64_t s) {
  return wrap(mx::round(a->array, static_cast<int>(decimals),
                        emily::resolve_stream(s)));
}
FINE_NIF(round, 0);

} // namespace
