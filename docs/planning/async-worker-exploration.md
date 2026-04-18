# Async Worker Exploration Plan

Exploration of dropping `WorkerThread::run_sync` in favour of an async
model where the NIF enqueues work onto the worker and the worker posts
the result back to the caller PID via `enif_send`.

## 0. Orientation

Destination model: lambda + caller PID + ref onto the worker queue;
worker thread calls `enif_send` with `{ref, reply}`; Elixir side has a
single `call/2` helper that does `ref = NIF(...); receive do {^ref, r} -> r end`.
Goal: raise effective concurrency from `dirty_cpu_schedulers` (default
= online cores) to approximately unbounded — bounded only by worker
queue depth and by the fact that each stream has one MLX thread.

### Ground truth from the repo

- **`c_src/emily_nif.cpp`** — only `to_binary` (line 93) and `eval`
  (line 116) are flagged `ERL_NIF_DIRTY_JOB_CPU_BOUND`. Every other
  NIF (`from_binary`, `shape`, `dtype`, `create_worker`, plus every op
  NIF in `c_src/ops/*.cpp`, and the memory NIFs in `c_src/memory.cpp`)
  is registered with flags `0` — i.e. runs on a **regular** scheduler.
  Today, all graph-construction ops already block regular schedulers
  via `run_sync`. That is the most urgent problem, because a blocking
  regular scheduler is far worse than a blocking dirty scheduler.
  - Individual graph-construction ops are meant to be <10 μs, so for
    the single-worker case the wall clock absorbed by blocking on a
    future is dominated by the hop-to-worker-thread cost, not MLX
    work. But under contention — N BEAM processes all pushing to the
    same worker — the NIF thread holds a regular scheduler slot while
    queued behind the worker. That is what the async model fixes.
- **WorkerThread queue** (`c_src/emily/worker.hpp`) is unbounded
  (`std::queue<std::function<void()>>`) with a condvar. Current
  behaviour: `run_sync` pushes a lambda, `cv_.notify_one()`, then
  blocks on a future.
- **Resource / binary return path** (`c_src/emily_nif.cpp:75-92`):
  `to_binary` creates a `mx::contiguous` copy on the worker, `eval`s
  it, then builds a resource binary in the caller's env. The
  `wrap(std::move(materialized))` runs in the caller's NIF thread
  (not the worker thread) — this matters for the async redesign,
  because the resource-binary construction has to be the very last
  step that runs with a valid env and the pinning resource alive.
- **Elixir side** (`lib/emily/backend.ex`): every op is
  `Native.foo(w, ref(t), …) |> wrap(out, w)`. The worker ref is read
  from the process dictionary via `worker()` → `Emily.MlxStream.default_worker/0`.
  No existing layer wraps NIF calls in `receive/4` — every `Native.*`
  is treated as a value-returning synchronous function.
- **`fine` semantics we rely on** (`deps/fine/c_include/fine.hpp`):
  - `ResourcePtr<T>` uses `enif_keep_resource` / `enif_release_resource`
    for refcount management; it is env-agnostic. Moving a
    `ResourcePtr` between threads is safe as long as those threads own
    their own reference (copy the ResourcePtr under a lock or by value
    before the thread owns it).
  - `make_resource` is env-agnostic — it calls `enif_alloc_resource`
    and does not require a scheduler env. We can `make_resource<Tensor>(…)`
    on the worker thread.
  - `enif_make_resource(env, ptr)` creates a term in a specific env.
    This IS env-bound.
  - `enif_make_resource_binary(env, ptr, data, size)` creates a binary
    term bound to a specific env and pins the resource via the `ptr`
    argument.
- **`FINE_NIF(name, flags)` macro** uses `static ERL_NIF_TERM name##_nif(...)`.
  For the async model we will either (a) keep returning a synchronous
  `ERL_NIF_TERM` (the ref), and have the worker `enif_send` later; or
  (b) drop `FINE_NIF` for the async path in favour of a hand-written
  NIF that returns only the ref.

---

## 1. Unknowns to resolve via spikes first

Each spike is a <1-day throwaway branch.

### Spike A — ErlNifEnv rules for cross-thread resource terms

**What to prove:** It is safe for the worker thread to:
1. build a `ResourcePtr<Tensor>` (via `make_resource`),
2. allocate an `ErlNifEnv* msg_env = enif_alloc_env()`,
3. construct the result term(s) in that env via `enif_make_resource(msg_env, ptr)`,
4. `enif_send(NULL, &caller_pid, msg_env, term)`,
5. `enif_free_env(msg_env)` if ERTS did not take ownership.

**Deliverable:** a minimal NIF `spike_async_tensor(w, shape)` that
builds a `Tensor` on the worker and posts it back. Verification is
a stress test, not a sanitizer run — ASan/TSan require rebuilding
the BEAM VM with matching flags, which is not practical here. Drive
16 parallel processes through a tight loop creating and discarding
1M tensors each. Success criteria:

- No crashes or hangs over the full run.
- `:erlang.memory(:total)` returns to baseline after
  `:erlang.garbage_collect/0` on every sender.
- `Emily.Native.get_active_memory()` (MLX allocator) returns to
  baseline — proves no tensor-resource leaks.
- Repeat the run N times; variance in end-state memory is zero.

**Specific API questions to resolve in-spike:**
- Does `enif_send(NULL, pid, msg_env, term)` transfer ownership of
  `msg_env` to ERTS (which frees it), or does the caller still free
  it? OTP docs say: with a non-null message env and a non-null caller
  env, ERTS transfers. With `NULL` caller env from a driver/non-scheduler
  thread, behaviour depends on OTP version. Verify on the OTP we ship
  against.
- `enif_self()` is documented as callable only on a scheduler thread.
  On the worker thread (non-scheduler) we MUST capture the PID at NIF
  entry and pass it via the lambda — do NOT call `enif_self` from the
  worker.
- Exception signalling: `enif_raise_exception` is a NIF-return
  construct; for async we need to send `{ref, {:error, {:exception, module, reason}}}`
  as a plain term. Spike confirms which atoms the Elixir-side
  `raise_error_with_message` equivalent translates to.

### Spike B — Resource binaries across env boundaries

**What to prove:** `enif_make_resource_binary` can be called from a
worker thread on an allocated env, will correctly pin the resource
for the lifetime of the BEAM binary, and the caller process's receive
loop observes a normal binary.

**Why uncertain:** the current `to_binary` returns a resource binary
built with the caller's NIF env. The alias is to MLX-owned memory
inside a `Tensor` resource. After `enif_send`, the binary term lives
in the receiving process's heap-fragment (ERTS copies terms on send),
but the resource refcount must be incremented before send and
decremented after the binary is GC'd. `enif_make_resource_binary` is
documented to do both; spike confirms.

**Deliverable:** a minimal `spike_async_to_binary(w, tensor)` that
returns a resource binary from an `enif_send` call. Write 1 GB of
aggregated binary data into `:binary.copy/1` + `:erlang.garbage_collect/1`
in the caller; observe MLX allocator `get_active_memory()` returns to
baseline. If it leaks, the pin lifetime is wrong and we need to
rebuild the binary in the caller's env.

**Backup plan if Spike B fails:** fall back to `enif_make_new_binary`
with a memcpy. For model inference the hot `to_binary` calls are
1–8 kB tokens; memcpy is fine. Only a few call sites —
`Nx.to_binary(big_weight_tensor)` — would regress, and those aren't
on any latency-critical path.

### Spike C — Per-process mailbox hygiene under batched ops

**What to prove:** the `receive do {^ref, r} -> r end` idiom scales
when N ops are queued concurrently from the same caller process. MLX
backend dispatches a single Nx op as dozens of Native calls. Each is
a separate `enif_send`. The caller's mailbox should never grow.

**Deliverable:** benchmark one process running 10k small-tensor ops
back-to-back. Check:
- `:erlang.process_info(self(), :message_queue_len)` stays at 0 or 1
  during sustained dispatch.
- Worst-case latency per op ≤ baseline `run_sync` latency + ~2 μs for
  the send/receive hop.

If mailbox grows, we have an ordering bug — the receive's `^ref` pin
should make this impossible, but if the worker ever sends out-of-order
we'd see it.

### Spike D — Queue depth and PID liveness

**What to prove:** `enif_send` to a dead PID is a silent no-op;
verify the worker does not block or leak on send-to-dead.

**Deliverable:** spawn a process, launch 100 async ops, kill it
mid-flight. Confirm:
- Worker drains its queue; no hang.
- The `Tensor` resources whose `enif_send` hit a dead PID still get
  released when the ResourcePtr in the worker's lambda-captured state
  goes out of scope (automatic via RAII).
- No growth in `mx::get_active_memory()` after `:erlang.garbage_collect/1`.

---

## 2. Phased implementation — four PRs

### Phase 1 — Async substrate (no behaviour change on the hot path)

**Scope:** Add the infrastructure to enqueue-and-post-back, without
removing `run_sync`. Pick one NIF as a canary.

**C++ changes:**

- `c_src/emily/worker.hpp`:
  - Add `template <typename F> void run_async(ErlNifPid caller, ERL_NIF_TERM ref_copy, F &&f)`.
    Enqueues a lambda that:
    1. Allocates `ErlNifEnv* msg_env = enif_alloc_env()`.
    2. Copies `ref_copy` into `msg_env` via `enif_make_copy` (the
       incoming ref term lives in the caller's NIF env, which is no
       longer valid by the time the worker runs).
    3. Runs `f(stream_, msg_env)` — returns an `ERL_NIF_TERM` tagged
       `{ref, {:ok, payload}}` or `{ref, {:error, reason}}`.
    4. Calls `enif_send(NULL, &caller, msg_env, term)`.
    5. Frees `msg_env` if `enif_send` didn't take ownership (OTP
       version-dependent — verify in Spike A).
  - Exception handling: wrap `f` in try/catch, build an error term in
    `msg_env` (mirroring `fine::nif_impl`'s catch ladder), send that.
  - Keep `run_sync` untouched — canary NIF flips to async, everything
    else still works.
- `c_src/emily/async.hpp` (new):
  - `make_ref(ErlNifEnv* env) → ERL_NIF_TERM` wrapping `enif_make_ref`.
  - Helpers to wrap a lambda: `emily_async(env, worker, [](Stream& s, ErlNifEnv* msg_env) { return wrap_term(...); })`
    that does the ref-making, pid-capture, enqueue, and return-ref.
- First async conversion: **`eval/2`** — already dirty-flagged, return
  type is `fine::Ok<>{}` (maps to `:ok`), trivial marshalling. Getting
  the synchronisation primitive right unlocks everything downstream.

**Elixir changes (`lib/emily/native.ex` + new `lib/emily/native/async.ex`):**

```elixir
def call(ref) do
  receive do
    {^ref, {:ok, result}} -> result
    {^ref, {:error, {:exception, kind, message}}} -> :erlang.raise(kind, message, [])
    {^ref, {:error, reason}} -> raise ArgumentError, inspect(reason)
  end
end
```

No default timeout — MLX ops can be long (seconds-scale `eval`); a
timeout needs to be policy at a higher layer. Optional `timeout` arg
only with use-case justification.

Only `eval/2` is converted in Phase 1:

```elixir
def eval(w, t), do: Async.call(eval_nif(w, t))
```

**Prove at this phase:**
- Full test suite passes with `eval` async. `to_binary` untouched.
- Crash test: kill a process while its `eval` is in flight; worker
  drains.
- Latency of a single `eval` on an already-resident tensor within 5%
  of `run_sync` baseline.

**Don't prove yet:** concurrency gains — with only `eval` converted,
nothing on the hot path is async.

### Phase 2 — Convert all op NIFs (tensor-returning)

**Scope:** Every NIF in `c_src/ops/*.cpp` returning a `ResourcePtr<Tensor>`
or a tuple of them. This is the bulk.

**C++ shape:** Helpers in `c_src/emily/async.hpp`:

```cpp
template <typename F>
ERL_NIF_TERM async_tensor(ErlNifEnv* env, ResourcePtr<WorkerThread> w, F&& f);

template <typename F>  // for multi-return
ERL_NIF_TERM async_tensor_tuple(ErlNifEnv* env, ResourcePtr<WorkerThread> w, F&& f);
```

where `F` is `mx::array(mx::Stream&)` or `std::tuple<mx::array, ...>(mx::Stream&)`.
Internally:
1. Build a ref with `enif_make_ref(env)`, encode as the NIF return.
2. Capture caller pid via `enif_self(env, &pid)`.
3. Push a lambda onto the worker that runs `f`, builds a
   `ResourcePtr<Tensor>` for each returned `mx::array` on the worker
   thread (safe — fine's `make_resource` is env-agnostic), then
   encodes `{ref_copy, {:ok, tensor_term}}` into `msg_env` via
   `enif_make_resource(msg_env, ptr)`, then `enif_send`.
4. Return the ref.

**Per-op rewrite (mechanical):**

```cpp
ERL_NIF_TERM add(ErlNifEnv* env,
                 ResourcePtr<WorkerThread> w,
                 ResourcePtr<Tensor> a,
                 ResourcePtr<Tensor> b) {
  return async_tensor(env, w, [a, b](mx::Stream& s) {
    return mx::add(a->array, b->array, s);
  });
}
```

Signature change: the NIF returns `ERL_NIF_TERM`, not
`ResourcePtr<Tensor>`. Fine's current `nif_impl` encodes the return
value; if we return a `fine::Term` it passes through verbatim. The
catch ladder around the NIF still runs in the NIF thread; synchronous
exceptions (bad tensor argument decoding) turn into exception-return
there. Async exceptions (MLX op failure on the worker thread) must be
caught inside the `run_async` lambda and sent as error terms.

**Conversion batches (order matters for diff review):**
1. `c_src/ops/unary.cpp` — 40 ops via `EMILY_UNARY` macro. Rewrite the macro.
2. `c_src/ops/binary.cpp` — similar, one macro change.
3. `c_src/ops/reduce.cpp`, `shape.cpp`, `cast.cpp`, `creation.cpp`,
   `sort.cpp`, `misc.cpp`, `index.cpp`.
4. Tuple-returning: `c_src/ops/linalg.cpp` (LU, SVD, QR, eigh),
   `quantize` → three-tuple, `random_split`.
5. Remaining: `fast.cpp`, `random.cpp`, `fft.cpp`, `pooling.cpp`,
   `conv.cpp`.
6. Core NIFs: `from_binary` (can stay sync — no worker needed),
   `shape` / `dtype` (no worker, pure reads — stay sync).

**Elixir changes:** `lib/emily/native.ex` stubs become
`def foo(w, a), do: Async.call(foo_nif(w, a))`. Dialyzer specs
unchanged — caller's return type is identical.

**Prove at this phase:**
- Full `mix test` (including `test/emily/backend_test.exs`) passes
  unchanged. The claim: async at the NIF layer is invisible above
  `Emily.Native`.
- Concurrency benchmark: rerun `test/soak/stream_concurrency_test.exs`
  (16 workers × 100 iters). Wall clock should drop if regular
  schedulers were the bottleneck.
- Memory neutrality: `test/soak/memory_test.exs` passes (MLX
  allocator returns to baseline).

### Phase 3 — Convert `to_binary` and decide resource-binary pinning strategy

**Scope:** The hardest case. `to_binary` currently runs
`mx::contiguous + mx::eval` on the worker and calls
`fine::make_resource_binary(env, pin, data, nbytes)` in the caller's
NIF env. The question is where (b) happens in the async world.

**Option 3a — Build resource binary on worker, send via msg_env:**
Spike B validates. Worker allocates msg_env, calls
`enif_make_resource_binary(msg_env, ptr, data, nbytes)`, wraps in
`{ref, {:ok, bin}}`, sends. ERTS's term-copy-on-send handles binary
descriptors specially: receiver gets a binary whose underlying
resource is ref-incremented. Fast path, no memcpy.

**Option 3b — Return resource handle async, build binary synchronously afterward:**
Worker sends `{ref, {:ok, tensor_resource}}`. Elixir `to_binary/2`:
```elixir
tensor_ref = Async.call(prepare_binary_nif(w, t))
Native.finalize_binary(tensor_ref)
```
Safe regardless of Spike B. One extra NIF hop (~1 μs).

**Option 3c — Fall back to memcpy (make_new_binary):**
Safe, simple, ~2 GB/s memcpy. Acceptable for everything except
multi-GB weight dumps.

**Decision gate:** Spike B outcome drives which ships. Default to 3a;
fall back to 3b if binary lifetime is not what we need. Measure all
three before committing.

**Prove at this phase:**
- Existing `Emily.Backend.to_binary/2` callers see no behaviour change.
- Chosen strategy's memcpy bandwidth matches sync baseline within 10%.
- `test/soak/zero_copy_roundtrip_test.exs` passes.

### Phase 4 — Backpressure and observability

**Scope:** Bounded worker queue with configurable limit and admission
policy. Separable from 1–3; can land later if measurements show no
real backpressure need.

**C++ changes:**
- `worker.hpp`: counted queue; `size_t max_depth_` (default:
  unbounded, configurable via env var at worker creation).
- On `run_async` enqueue: if queue size ≥ max_depth, either
  (a) reject — send `{ref, {:error, :overload}}` synchronously; or
  (b) block the NIF thread in `cv_.wait` — restores old-world
  throttling. Option (a) is more BEAM-idiomatic (let higher layers
  retry) but surprises Nx. Option (b) keeps sync-model semantics.

**Elixir changes:**
- `lib/emily/stream.ex`: `new/1` accepts `max_queue_depth: n`.
- `Emily.Native.Async.call/1` on `:overload` raises
  `Emily.OverloadError`.
- Telemetry: `[:emily, :worker, :enqueue]` / `[:emily, :worker, :dequeue]`
  spans; gauge for queue depth.

**Prove at this phase:**
- 10k-op burst from one process: peak queue depth bounded; wall-clock
  unchanged (throttling is desired under sustained pressure).
- Telemetry produces a depth time-series.

---

## 3. Answers to the numbered questions

### 1. Result marshalling across `enif_send`

- `fine::ResourcePtr<T>` is env-agnostic and refcounts safely across
  threads. Building it on the worker is fine. `make_resource` calls
  `enif_alloc_resource` which has no env dependency.
- `enif_make_resource(env, ptr)` is env-bound. For async send, `env`
  must be a message env allocated via `enif_alloc_env()`. Terms from
  the calling NIF's env are invalid as soon as the NIF returns; we
  MUST NOT capture those in the lambda. Copy incoming non-resource
  terms (e.g., the ref) with `enif_make_copy(msg_env, caller_term)`
  before enqueuing.
- `enif_send(NULL, &pid, msg_env, term)` on OTP 24+ with non-null
  `msg_env` transfers ownership to ERTS for freeing — do NOT call
  `enif_free_env`. Verify in Spike A.
- Multiple resource terms in one message (e.g., SVD's 3-tuple):
  allocate one msg_env, make each resource term, build the tuple in
  msg_env, send. All three resource refcounts are incremented by
  `enif_make_resource` and decremented at GC on the receiver side —
  correct by construction.

### 2. `to_binary` specifically

Three viable strategies with Spike B dependency:
- **3a — Cross-env resource binary:** Fast path, no memcpy. Requires
  Spike B confirmation.
- **3b — Two-step:** Async returns materialised-tensor ref; a second
  sync NIF builds the resource binary in caller's env. One extra NIF
  hop.
- **3c — Memcpy via `enif_make_new_binary`:** Unconditional memcpy.
  ~2 GB/s cap. Fine for anything under ~100 MB.

Recommend 3a if Spike B is green, 3c if not. Skip 3b unless 3a fails
and memcpy shows up in profiles.

### 3. Backpressure

A burst of 10 000 ops from one process into an unbounded queue has
two concerns:
- **MLX graph memory:** each lazy `mx::array` holds refs to its
  parents. A 10k-op graph that hasn't been `eval`'d holds
  GPU-buffer refs for every leaf. Sync model implicitly throttles
  (can't submit N+1 until N is on the queue and dequeued).
- **Worker queue memory:** each `std::function<void()>` ~100–300
  bytes. 100k deep = ~20 MB; negligible.

**Recommendation:** default unbounded in Phase 1–3 (measure first),
add bounded-queue option in Phase 4 with **block-on-enqueue** default
when bounded. Blocking preserves sync-model backpressure semantics.
Rejection (`:overload`) is surprising; opt-in only.

### 4. Failure modes (caller dies mid-flight)

- **`enif_send` to dead pid:** documented silent drop. Safe. The
  worker's lambda holds the only strong refs to result tensors; when
  the lambda returns, ResourcePtrs go out of scope, MLX buffers free.
  No leak.
- **Caller dies before NIF returns the ref:** impossible — the NIF is
  synchronous from the caller's view up to enqueue.
- **Worker thread C++ exception:** already caught in `run_sync`.
  Mirror in `run_async`: catch → encode error term → send.
- **Worker shutdown with items in flight:** `WorkerThread::stop()`
  currently drains before joining. Add `if (stop_) { send error(:shutdown); return; }`
  at enqueue time to reject post-stop enqueues.
- **PID monitoring:** do NOT `enif_monitor_process` per call.
  `enif_send`'s silent-drop semantics suffice. Skip monitoring.

### 5. Measurement

Three benchmarks, in order of increasing realism:

1. **Micro: NIF hop latency.** One process, back-to-back
   `Native.add(w, a, b)`. Async within 3× sync (add ~2 μs for
   send+receive+enif_alloc_env). `:timer.tc` over 100k iterations.

2. **Op-level concurrency: shared worker, many processes.** Adapt
   `test/soak/backend_concurrency_test.exs` to pin all workers to a
   single shared `Emily.Stream`. Sync today saturates regular
   schedulers; async should unclog them. Metric: aggregate wall-clock
   / p50 / p99.

3. **End-to-end: N concurrent Qwen3 decodes, per-process streams.**
   Extend `bench/qwen3_tokens_per_sec.exs`: spawn N=1,2,4,8,16
   decoding tasks, each `Emily.Stream.with_stream`'d. Report aggregate
   tokens/sec. Success: async at N=16 ≥ 1.5× sync at N=16.

Additional invariants: `test/soak/eval_concurrency_test.exs` and
`backend_concurrency_test.exs` must pass unchanged.

### 6. Testing

**Existing suite:** ~95% of `test/emily/**` is backend-behaviour
tests — operate above `Emily.Native`, should pass unchanged. Two
classes need attention:
- `test/emily/native_test.exs`: exercises NIFs directly. Add
  assertions that:
  - NIF returns a reference synchronously.
  - Ref is consumed by `Async.call/1` to produce tensor / binary /
    atom.
  - Malformed arg raises at decode time (synchronously, before
    enqueue).
  - Runtime MLX error raises on `Async.call` (asynchronously,
    message-delivered error).
- `test/soak/*`: already exercise concurrency. Strictly more forgiving
  in async world. Re-run with `async: true` modules where possible.

**New tests:**
- `test/emily/async_test.exs`:
  - `call/1` pattern-matches the right ref (ordering safety).
  - Unused ref does not leak mailbox messages.
  - Process killed mid-flight — worker survives, next caller works.
  - Worker's PID capture happens at enqueue time, not at dequeue
    (caller may move scheduler threads).
- Property test (`stream_data`): enqueue N ops, assert all N replies
  received, one per ref, any order.

**Timing-sensitive:** `test/emily/telemetry_test.exs` measures
`:telemetry.span` timings. Still wraps full `Async.call`; same
semantics. Verify.

### 7. Incremental migration

**Yes, one-NIF-at-a-time.** Boundary is `Emily.Native.foo/n` — from
`Emily.Backend`'s perspective, still synchronous whether implemented
as `nif()` or `Async.call(foo_nif(...))`.

- Each op NIF can flip independently. Elixir caller untouched.
- `create_worker`, `shape`, `dtype`, `from_binary` — don't need
  conversion.
- `to_binary` and `eval` — Phase 1 / Phase 3 handle explicitly.
- Every other op NIF — Phase 2, in batches.

Mixed sync/async during conversion is fine for correctness but
misleading for profiling. Convert the full `c_src/ops/unary.cpp`
batch or none — mid-conversion profiles attribute latency weirdly.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Spike B fails — resource-binary cross-env send doesn't pin correctly | Fall back to 3c (memcpy). No benchmarks degrade meaningfully. |
| `enif_send` semantics differ on OTP <25 | Pin OTP version for async feature; document min OTP in CHANGELOG. |
| Downstream caller relies on strict op ordering within a single process | Ordering preserved by `receive ^ref`; caller sees strictly sequential returns. Not actually at risk. |
| Hidden BEAM process dict dependency | `worker()` read happens at NIF-dispatch time from Elixir; survives async. |
| Regression: async adds ~2 μs per op; sub-μs Nx pipelines slow down | Microbenchmark #1. If regression on small ops (e.g., `shape`) is unacceptable, keep them sync — no reason graph-introspection needs async. |
| Worker thread becomes the new bottleneck | Already true under sync. No change. Multi-worker within one stream is out of scope (MLX's encoder is thread-local). |

---

## 5. Punch list — first actions before any implementation code

1. **Spike A** (ErlNifEnv rules) — 2–4 h.
2. **Spike B** (resource-binary cross-env) — 2–4 h. If red, document
   and move to 3c fallback.
3. **Spike C** (mailbox hygiene under load) — 1 h.
4. **Spike D** (dead-pid robustness) — 30 min.
5. **Only after A+B+C+D are green:** start Phase 1 PR.

After spikes, each phase is a standalone PR:
- **Phase 1 PR:** `async.hpp` helpers + `eval` conversion + new
  `Emily.Native.Async.call/1` + basic tests. ~400 LOC diff.
- **Phase 2 PR:** bulk op conversion across `c_src/ops/*.cpp` +
  `native.ex` stub regeneration. ~800 LOC diff, mostly mechanical.
- **Phase 3 PR:** `to_binary` async with chosen strategy (3a
  preferred). ~150 LOC.
- **Phase 4 PR** (optional, later): bounded queues + telemetry.
  ~300 LOC.

Total estimated effort: 3–5 engineer-days across the four PRs,
assuming spikes clear.
