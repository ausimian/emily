# Emily vs EXLA — performance comparison & recommendations

A like-for-like benchmark of the local (unpublished) **Emily** backend against
the **EXLA** backend, with an analysis of where Emily is slower and concrete
recommendations to close those gaps.

Harness: [`bench/emily_vs_exla.exs`](emily_vs_exla.exs). Raw machine-generated
numbers: [`bench/emily_vs_exla_results.md`](emily_vs_exla_results.md). Re-run
with `elixir bench/emily_vs_exla.exs` (the harness pins the local Emily via
`{:emily, path: ".."}`).

---

## 0. Read this first — what is actually being compared

On Apple Silicon, **EXLA has no GPU target**. The only PJRT client it ships for
`macos-arm64` is the **host / CPU** client, which is what ran here
(`exla client : host (1 device)`). Emily runs on the **Metal GPU** via MLX.

So this is **GPU (Emily/MLX) vs CPU (EXLA/XLA-host)** — not GPU-vs-GPU. The two
backends have completely different cost structures:

| | Emily (MLX, GPU) | EXLA (XLA, CPU) |
| --- | --- | --- |
| Fixed per-op latency | High (~160–280 µs floor: BEAM↔worker hop + Metal command-buffer commit + GPU sync) | Low (~80–110 µs: a CPU kernel call) |
| Throughput at scale | High (thousands of GPU ALUs) | Moderate (12 CPU cores, AMX/NEON) |
| Small-kernel efficiency | Low (per-kernel launch latency dominates) | High (cache-resident, no launch cost) |

That difference explains every result below: **CPU wins when the work is small
(latency-bound); GPU wins when the work is large (throughput-bound).** The two
questions for Emily are (a) *where* the crossover sits and (b) whether Emily's
per-op/per-kernel overhead is larger than it needs to be.

A second cross-cutting fact, measured this run via a telemetry counter:
**every model lowered fully native on Emily — zero `BinaryBackend` fallbacks and
zero native→evaluator fallbacks across all four model tiers.** So none of
Emily's losses below are coverage gaps; they are all kernel/dispatch efficiency.

### Environment

| Field | Value |
| ----- | ----- |
| Date | 2026-06-07 |
| Host | Apple M4 Pro, 12 cores, 24GB memory |
| Elixir / OTP | 1.19.5 / 28 |
| Emily | 0.6.1 (local checkout, AOT MLX variant) |
| EXLA / XLA | 0.12.0 / 0.10.0 (host / CPU client) |
| Nx | 0.12.1 |

Lanes: **exla** (`compiler: EXLA`, CPU) · **emily-eager** (`Emily.Backend`
under `Nx.Defn.Evaluator`, op-by-op GPU dispatch) · **emily-native**
(`Emily.Compiler, native: true`, whole graph replayed in one NIF call) ·
**emily-fuse** (native + `fuse: true`, `mx::compile` kernel fusion).

> **Quantization was dropped on purpose.** Emily's int4/int8 inference is MLX
> `quantized_matmul` over `Emily.QuantizedWeight`; EXLA has no equivalent
> quantized kernel, so there is no like-for-like lane to put beside it. A
> "quantized Emily vs f32 EXLA" race would measure quantization, not the
> backends. It belongs in a separate Emily-only quantization benchmark.

---

## 1. Executive summary

* **Emily wins the workloads it targets — GPU-friendly model inference.**
  Through its native compiler it beats EXLA-CPU on Qwen3-0.6B decode **1.67×**
  (68.8 vs 41.1 tok/s) and ViT-base **2.2×** (25.2 vs 55.4 ms); DistilBERT QA is
  ~parity (**1.06×**). Large matmul is **5.0×** (2048², 3.5 vs 17.4 ms).
* **Emily loses badly on small-kernel-bound work.** The same per-op penalty that
  makes small tensors 1.8–2.3× slower than EXLA (§2) scales up to **Whisper-tiny
  being ~11× slower** (979 vs 86 ms) — the worst result in the suite — and makes
  the **eager backend 3.5× slower on Qwen3 decode** (§4).
* **The deciding factor is per-op tensor size, exactly as the microbench
  predicts.** ViT-base (hidden dim 768, fat matmuls) is in the GPU's favour;
  Whisper-tiny (dim 384, a 30 s mel/encoder path made of many tiny kernels) is
  not. Both lower fully native — the gap is kernel efficiency, not coverage.
* **`fuse` is workload-dependent, not a universal add-on.** Best lane on Qwen3
  (cached loop body) and DistilBERT; neutral-to-slightly-worse on ViT; no help
  on Whisper.

The places Emily trails EXLA, by impact: **(1)** Whisper-tiny / small-kernel
models, **(2)** the per-op dispatch/sync floor on small tensors, **(3)** the
eager backend on decode loops. Recommendations in §7.

---

## 2. Tier 1 — op microbenchmarks

Mean µs/call, **lower is better**. Each call computes `op(...) |> Nx.sum()` and
reads back the scalar, so every op is fully realized. The `Emily/EXLA` column is
`best-emily-lane ÷ exla`: **< 1.0 = Emily faster**, **> 1.0 = Emily slower**.

| op | size | exla (CPU) | emily-eager | emily-native | emily-fuse | Emily/EXLA | winner |
| -- | ---- | ---------: | ----------: | -----------: | ---------: | ---------: | ------ |
| add | 256 | **95.7** | 248.3 | 231.2 | 217.6 | 2.27× | EXLA |
| add | 1024 | **229.2** | 432.2 | 448.4 | 371.1 | 1.62× | EXLA |
| add | 4096 | 2980.2 | 1324.2 | 1343.3 | 1385.5 | 0.44× | Emily 2.3× |
| mul | 256 | **106.5** | 229.6 | 209.2 | 237.0 | 1.96× | EXLA |
| mul | 1024 | **223.3** | 448.7 | 341.3 | 369.0 | 1.53× | EXLA |
| mul | 4096 | 2914.6 | 1371.2 | 1481.2 | 1417.0 | 0.47× | Emily 2.1× |
| exp | 256 | **107.2** | 215.1 | 226.4 | 202.1 | 1.89× | EXLA |
| exp | 1024 | **320.6** | 340.3 | 348.7 | 351.6 | 1.06× | ~tie |
| exp | 4096 | 3661.1 | 1216.6 | **1106.0** | 1155.8 | 0.30× | Emily 3.3× |
| sum | 256 | **79.2** | 279.6 | 180.8 | 207.4 | 2.28× | EXLA |
| sum | 1024 | **195.9** | 218.0 | 256.0 | 268.0 | 1.11× | EXLA |
| sum | 4096 | 2011.9 | 752.4 | 718.1 | **635.0** | 0.32× | Emily 3.2× |
| softmax | 256 | **136.3** | 335.9 | 286.4 | 245.8 | 1.80× | EXLA |
| softmax | 1024 | **461.0** | 596.9 | 477.3 | 539.5 | 1.04× | EXLA |
| softmax | 4096 | 5230.0 | 2734.0 | 2707.9 | **2134.1** | 0.41× | Emily 2.5× |
| matmul | 128 | **98.3** | 197.1 | 207.6 | 210.0 | 2.01× | EXLA |
| matmul | 512 | **452.7** | 558.9 | 581.8 | 594.8 | 1.23× | EXLA |
| matmul | 1024 | 2488.1 | **744.3** | 809.0 | 811.8 | 0.30× | Emily 3.3× |
| matmul | 2048 | 17409.7 | 3523.3 | **3508.4** | 3520.5 | **0.20×** | Emily 5.0× |

**The crossover.** Every op family flips between ~512 and ~2048 elements/dim:
EXLA-CPU wins small, Emily-GPU wins large. matmul is the sharpest — EXLA wins to
512, Emily wins 3.3× at 1024 and **5.0× at 2048**. Below the crossover, Emily's
realized-single-op floor (~160–280 µs regardless of op — pure dispatch + sync,
not compute) loses to EXLA's ~80–110 µs CPU floor. **That floor is the root of
every Emily loss in this report.** Keep it in mind for §5 and §6.

---

## 3. Tier 2 — DistilBERT QA (one encoder forward)

`distilbert-base-uncased-distilled-squad`, mean µs/call over 3 runs. **Lower is
better.** Zero fallbacks on every Emily lane.

| lane | ms/call | vs EXLA |
| ---- | ------: | ------: |
| exla (CPU) | 8.60 | 1.00× |
| emily-eager (GPU) | 17.32 | 0.50× (slower) |
| emily-native (GPU) | 12.42 | 0.69× (slower) |
| **emily-fuse (GPU)** | **8.15** | **1.06× (faster)** |

Roughly parity. DistilBERT (hidden dim 768, 6 layers, ~6-token QA input) sits
right at the crossover. `fuse` is the only Emily lane that edges past EXLA; the
plain `native` lane is noisy here (per-run 14.9 → 9.5 ms as MLX warms) and lands
slower. Eager pays the per-op floor across the whole encoder and loses 2×.

---

## 4. Tier 3 — Qwen3-0.6B greedy decode (48 new tokens)

Tokens/sec over 3 runs. **Higher is better.** Zero fallbacks on every Emily lane.

| lane | tok/s | vs EXLA |
| ---- | ----: | ------: |
| exla (CPU) | 41.1 | 1.00× |
| emily-eager (GPU) | 11.83 | **0.29× (3.5× slower)** |
| emily-native (GPU) | 63.0 | 1.53× (faster) |
| **emily-fuse (GPU)** | **68.78** | **1.67× (faster)** |

`fuse` wins (1.67×): the per-token forward is the loop body, the `mx::compile`'d
callable is cached per stream, and the compile cost amortizes over every step.
**emily-eager (0.29×) is the second-worst result in the suite** — autoregressive
decode issues thousands of tiny ops per token, each paying the §2 floor. The
5.8× gap between eager (11.8) and fuse (68.8) on identical hardware is the cost
of per-op dispatch, made visible.

---

## 5. Tier 4 — ViT-base image classification (one forward)

`google/vit-base-patch16-224`, synthetic 224×224 image, mean µs/call over 3
runs. **Lower is better.** Zero fallbacks on every Emily lane.

| lane | ms/call | vs EXLA |
| ---- | ------: | ------: |
| exla (CPU) | 55.44 | 1.00× |
| emily-eager (GPU) | 43.31 | 1.28× (faster) |
| **emily-native (GPU)** | **25.22** | **2.20× (faster)** |
| emily-fuse (GPU) | 26.02 | 2.13× (faster) |

**Emily wins every lane — even eager beats EXLA.** This is the mirror image of
Whisper (§6) and the proof that model size, not model *kind*, decides the
outcome. ViT-base has hidden dim 768 and runs 197 patches through fat
`768×768`/`768×3072` matmuls — squarely in the GPU-favourable regime from §2
(matmul ≥ 1024). The native compiler doubles EXLA; `fuse` adds nothing here (one
forward, no loop body to reuse — cf. §7.3).

---

## 6. Tier 5 — Whisper-tiny transcription (≤25 new tokens)

`openai/whisper-tiny`, ~1 s synthetic audio (padded to the 30 s mel window),
mean µs/call over 3 runs. **Lower is better.** Zero fallbacks on every Emily lane.

| lane | ms/call | vs EXLA |
| ---- | ------: | ------: |
| exla (CPU) | 85.6 | 1.00× |
| emily-eager (GPU) | 1913.9 | **0.045× (22× slower)** |
| emily-native (GPU) | 986.2 | **0.087× (11× slower)** |
| emily-fuse (GPU) | 979.3 | **0.087× (11× slower)** |

**This is Emily's worst result, and the most instructive.** The natural guess —
the mel STFT's `fft` falling back to `BinaryBackend` — is **wrong**: the
fallback counter read **binary=0, compiler=0** on every lane. Whisper lowers
fully native (the conformance suite proves the same with `native_fallback:
:raise`); it is simply ~11× slower on the GPU than on the CPU.

Why? Whisper-tiny is *tiny* — hidden dim **384**, 4 encoder + 4 decoder layers —
but it always runs the full **30 s / 1500-position** mel + encoder path plus a
conv frontend and an STFT over 3000 frames. That is a very large number of
**small** kernels. Small kernels are exactly where the GPU's per-launch latency
dominates and its parallelism goes unused (the matmul-128 / small-elementwise
regime of §2), while the CPU runs them cache-resident with no launch cost. The
native compiler removes BEAM-side dispatch (eager → native is 1.9×) but cannot
remove the thousands of small Metal kernel launches underneath, so it still
trails EXLA 11×.

This is the single most valuable signal in the report: **a fully-supported model
can still be an order of magnitude slower on Emily purely because of small-kernel
inefficiency.** See §7.1.

---

## 7. Where Emily is slower than EXLA — analysis & recommendations

### 7.1 Small-kernel-bound models (highest impact — Whisper, eager decode)

**Symptom.** Whisper-tiny native is 11× slower than EXLA; eager Qwen3 decode is
3.5× slower; every op ≤ 1024/dim is 1.5–2.3× slower. All fully native, zero
fallbacks — it is kernel/launch efficiency, not coverage.

**Root cause.** A model built from many small tensors (small hidden dim and/or
many positions/steps) issues a flood of small Metal kernels. Each has a fixed
launch + sync latency and underuses the GPU. The native compiler collapses the
*BEAM-side* dispatch into one NIF call but does not coalesce the *GPU-side*
kernels, so the floor survives.

**Recommendations.**

1. **Kernel fusion is the highest-leverage lever — extend it past the loop
   body.** `fuse` already cuts Qwen3 by reusing a fused decode body. Whisper
   gets nothing because its cost is the *encoder + featurizer*, not a reused
   loop. Investigate fusing the encoder/featurizer elementwise+norm runs (the
   same RMSNorm/softmax/residual chains `fuse` already targets) so a single
   forward emits fewer, larger kernels. The §2 result that `fuse` wins softmax
   at 4096 (2134 vs 2708 native) shows the mechanism works when there's
   something to fuse.
2. **Profile Whisper into featurizer vs encoder vs decode.** The STFT (3000
   frames, n_fft 400) and the conv frontend are the prime suspects for small,
   serial kernels. `[:emily, :eval, *]` spans plus MLX's own profiler will say
   whether to batch the STFT (one big FFT/conv vs many small ones) or the
   encoder attention. Batching many small FFTs into one call is the classic fix.
3. **Shrink the per-kernel floor itself (helps everything).** Even `native` op
   at 256 sits at ~180 µs. Split that floor — NIF round-trip vs Metal commit vs
   GPU wait — and attack the largest slice (e.g. an inline fast-path for trivial
   programs that skips the worker hop). Lowering the floor pulls the §2 crossover
   down and lifts Whisper, eager decode, and all small-tensor work at once.
4. **Document the size dependence.** State plainly: *Emily's GPU advantage starts
   around hidden-dim ≥ 768 / matmul ≥ 1024; small models (Whisper-tiny, dim 384)
   or per-op-synchronous loops can be slower than EXLA-CPU.* ViT (2.2× faster)
   vs Whisper-tiny (11× slower) is the headline contrast to lead with.

### 7.2 Per-op dispatch/sync floor on small tensors

**Symptom.** Every op ≤ 1024/dim is 1.0–2.3× slower; Emily's realized-single-op
floor is ~160–280 µs vs EXLA's ~80–110 µs.

**Root cause.** Realizing one op costs a synchronous BEAM↔worker hop, a Metal
commit, and a GPU completion wait — pure latency, roughly fixed per realized op.

**Recommendations.** Same levers as §7.1.3 (shrink the floor) plus: **batch eager
dispatch** — coalesce a run of pending lazy ops into one worker submission before
the next forced sync, so the eager backend stops paying one round-trip per op.
That directly attacks the eager-lane losses in §3, §4, §6.

### 7.3 `fuse` is workload-dependent, not free

**Symptom.** `fuse` is the best lane on Qwen3 and DistilBERT, neutral on ViT, and
no help on Whisper; on DistilBERT in the *first* run it even regressed vs native
(per-call `mx::compile` cost not amortized over a single forward).

**Root cause.** `mx::compile` has a per-callable build cost that pays off only
when the fused callable is reused — many times in a decode loop, never within a
single forward.

**Recommendations.**

1. **Gate `fuse` on reuse.** The compiler knows whether the traced Expr contains
   a `defn while` / repeated body; default `fuse` on for those and off for
   one-shot forwards, instead of relying on the caller to guess.
2. **Persist the fused-callable cache across `Nx.Serving.run/2`.** If a serving
   recompiles per request, a forward-only model never amortizes the build —
   verify the per-stream cache key survives between calls.
3. **Document the rule of thumb:** *`fuse` helps decode loops (cached body); it
   is neutral-to-negative on single forward passes.* The README currently
   presents it as a strict add-on to native, which §3 (win), §5 (neutral), and
   §6 (no help) show it is not.

### 7.4 Eager backend on autoregressive decode

**Symptom.** `Emily.Backend` under the evaluator does Qwen3 decode at 11.8 tok/s
vs EXLA 41.1 and Emily-native 63.0 — 3.5× slower than EXLA.

**Recommendation.** Make the native compiler the documented, signposted default
for generation: emit a one-time `Logger.warning` / telemetry event when a
`defn while` decode loop is about to run under the plain evaluator on
`Emily.Backend`, and treat eager as a correctness/compat path, not a perf path.
The 5.8× eager→fuse gap is too large to leave to chance.

---

## 8. Bottom line

Emily delivers on its purpose for **GPU-friendly inference**: through the native
compiler it beats EXLA-CPU by 1.06–2.2× on DistilBERT/Qwen3/ViT and up to 5.0× on
large matmul. Its weaknesses are all one phenomenon — a **per-op/per-kernel
latency floor ~2–3× EXLA's** — surfacing at three scales: small tensors (§2),
small-kernel models (Whisper-tiny, **11× slower**, §6), and per-op-synchronous
eager decode (§4). Crucially, **none of it is a coverage gap**: every model
lowered fully native with zero fallbacks, so the entire opportunity is kernel
efficiency. The ranked fixes — extend fusion beyond the decode-loop body (§7.1.1),
profile and batch Whisper's small-kernel front end (§7.1.2), shrink the per-kernel
floor (§7.1.3), make `fuse` reuse-aware (§7.3), and steer generation onto the
native compiler (§7.4) — would remove every case in this suite where Emily trails
EXLA.

> **Caveat.** EXLA here is CPU-only (no GPU client on macOS arm64), so this
> compares MLX-on-GPU against XLA-on-CPU. It is the comparison available on this
> platform and the one most Elixir-on-Apple-Silicon users face when choosing a
> backend, but it is not a kernel-quality comparison of MLX vs XLA on equal
> hardware. Numbers are single-host, f32, AOT MLX variant, 3 measured runs after
> a warm-up. Re-run with `elixir bench/emily_vs_exla.exs`.
