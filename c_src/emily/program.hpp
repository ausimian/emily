// Program: a compiled, replayable Nx.Defn.Expr graph for the Expr->MLX
// single-NIF compiler.
//
// Built once by `compile_program` (parses the flat IR and captures
// strong refs to weight/const tensors); replayed per call by
// `eval_program` on the worker thread, which rebuilds the `mx::array`
// DAG from the cached instruction list + fresh inputs and evals it.
//
// The win: one BEAM<->worker NIF round-trip per *invocation* instead of
// one per op. Weights cross the boundary once (here) and are never
// re-serialized per eval. See c_src/program.cpp and lib/emily/program.ex.

#pragma once

#include "opcodes.hpp"
#include "tensor.hpp"
#include "worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace emily {

namespace mx = mlx::core;

// Operand references are packed into an int64 by the Elixir lowerer
// (`Emily.IR.pack_ref/1`): the high bits carry the slot kind, the low
// bits the index. Unpacked here during compile-time validation and
// during replay. Keep in sync with lib/emily/ir.ex.
namespace ref {

inline constexpr int kTagShift = 48;
inline constexpr int64_t kIndexMask = (int64_t(1) << kTagShift) - 1;

enum class Kind : int64_t { Input = 0, Capture = 1, Const = 2, Instr = 3 };

inline Kind kind_of(int64_t r) {
  return static_cast<Kind>((r >> kTagShift) & 0x3);
}

inline int64_t index_of(int64_t r) { return r & kIndexMask; }

} // namespace ref

class Program;

struct CompiledInstr {
  Opcode opcode;
  std::vector<int64_t> operands;             // packed refs
  std::vector<std::vector<int64_t>> iattrs;  // integer attrs (shapes/axes/dtype codes)
  // Nested programs an instruction carries (empty for all but control
  // flow). `while` holds [condition, body]; each is replayed with the
  // loop-carried state bound as its inputs (`{:input, i}` -> state[i]).
  // Held by ResourcePtr so the child program resources stay alive for the
  // parent's lifetime (refcounted, like every capture).
  std::vector<fine::ResourcePtr<Program>> subprograms;
};

// One resource per compiled program. `captures` / `consts` hold strong
// BEAM refs so the weight buffers stay alive for the program's lifetime
// (fine's ResourcePtr bumps the refcount exactly like every op capture).
class Program {
public:
  int64_t n_inputs = 0;
  std::vector<fine::ResourcePtr<Tensor>> captures;
  std::vector<fine::ResourcePtr<Tensor>> consts;
  std::vector<CompiledInstr> instrs;
  std::vector<int64_t> outputs; // packed refs

  using CompiledFn =
      std::function<std::vector<mx::array>(const std::vector<mx::array> &)>;

  // One `mx::compile`d replay callable, plus a weak handle to the worker
  // whose thread-local compiler cache holds its traced graph. The handle
  // lets `~Program` drop the callable back on that worker thread (see the
  // destructor) — `mx::compile`'s cache erase is thread-affine.
  struct CompiledEntry {
    std::weak_ptr<State> worker;
    CompiledFn fn;
  };

  // CM6: opt-in mx::compile cache. One compiled replay callable per stream
  // index — the compiled graph bakes in the captured weights and the
  // stream, so it must be keyed by stream and rebuilt if used on a
  // different one. Built lazily on the first compiled eval (eval_mode 3).
  // This is the *secondary* encode win; the main dispatch-collapse win is
  // the single-NIF replay itself.
  std::mutex compile_mtx;
  std::map<int, CompiledEntry> compiled;

  Program() = default;

  // Drop each compiled callable on the worker thread that built it. MLX's
  // compiler cache is `thread_local` and the callable's deleter calls
  // `compile_erase` wherever it is destroyed; this resource is collected on
  // a BEAM/GC thread, so destroying the callable here would erase the wrong
  // thread's cache — leaking the worker's traced graph (and its refs to the
  // captured weight buffers), and risking a later `fun_id` (a recycled heap
  // address) colliding with the stale entry. Posting the drop to the worker
  // makes the erase land on the right cache. If the worker is already gone,
  // its thread exit destroyed the thread-local cache (and our entry) for us.
  ~Program() {
    for (auto &kv : compiled) {
      CompiledEntry &entry = kv.second;
      if (!entry.fn) {
        continue;
      }
      if (auto st = entry.worker.lock()) {
        post_to_worker(*st, [fn = std::move(entry.fn)]() mutable { fn = nullptr; });
      }
    }
  }

  // Movable/copyable would be wrong (std::mutex member), and the explicit
  // destructor suppresses the implicit moves anyway; spell it out.
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;
};

} // namespace emily
