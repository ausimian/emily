// Pure op cores: the `mx::array`-building expression for each Emily op,
// free of NIF / async / enif plumbing. One source of truth per op,
// called from BOTH the eager per-op NIF (c_src/ops/*.cpp) and the
// Expr-compiler program replay (c_src/program.cpp). Keeping the core
// here means the compiled (single-NIF) path can never numerically drift
// from the eager path — they invoke the same function.
//
// CM0 ships only `add` (the prototype op). CM1 fills in the full
// primitive set as each c_src/ops/*.cpp op is split into core + thin NIF.

#pragma once

#include <mlx/mlx.h>

namespace emily::ops {

namespace mx = mlx::core;

// --- Binary elementwise ---

inline mx::array add_core(const mx::array &a, const mx::array &b,
                          mx::Stream &s) {
  return mx::add(a, b, s);
}

} // namespace emily::ops
