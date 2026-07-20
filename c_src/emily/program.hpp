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
  // Value-initialized so a default-constructed instr has a defined opcode
  // (the other members are containers that self-initialize). Every real
  // construction aggregate-initializes all fields, overriding this.
  Opcode opcode{};
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
  //
  // `mutable`: the cache is pure memoization, so it is filled even through a
  // `const Program &` — the `while` arm of the replay compiles its *child*
  // (body) program, which it holds by const reference (CM14 fused-while).
  mutable std::mutex compile_mtx;
  mutable std::map<int, CompiledEntry> compiled;

  Program() = default;

  // Drop each compiled callable on the worker thread that built it. MLX's
  // compiler cache is `thread_local` and the callable's deleter calls
  // `compile_erase` wherever it is destroyed; this resource is collected on
  // a BEAM/GC thread, so destroying the callable here erases the wrong
  // thread's cache. Posting the drop to the worker makes the erase land on
  // the right cache, so `post_to_worker` moves the callable into the queued
  // task only once the worker accepts it.
  //
  // If the worker is stopping (or already gone) the post is declined and the
  // callable is destroyed here on the GC thread instead — which is benign:
  //   * the wrong-thread `compile_erase` is a no-op (the GC thread's cache
  //     never held this `fun_id`);
  //   * the worker's thread exit clears its thread_local compiler cache,
  //     reclaiming the real traced graph and its captured weight refs; and
  //   * the recycled-`fun_id` collision the on-worker erase guards against
  //     can't fire on a stopping worker — it runs no further compiles.
  // So on the declined path we simply let `drop` destruct below (issue #172).
  //
  // post_to_worker constructs a std::function (may throw bad_alloc); a throw
  // from this best-effort cleanup destructor is unrecoverable and would
  // std::terminate regardless, so the escape is accepted here.
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~Program() {
    for (auto &kv : compiled) {
      CompiledEntry &entry = kv.second;
      if (!entry.fn) {
        continue;
      }
      if (auto st = entry.worker.lock()) {
        std::function<void()> drop = [fn = std::move(entry.fn)]() mutable {
          fn = nullptr;
        };
        post_to_worker(*st, drop);
      }
    }
  }

  // Movable/copyable would be wrong (std::mutex member), and the explicit
  // destructor suppresses the implicit moves anyway; spell it out.
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;
};

} // namespace emily
