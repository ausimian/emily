# M6 de-risk — `mlx::core::compile` microbenchmark

Phase-1 measurement for PLAN milestone M6. The PLAN gates the whole
milestone on a ≥20% win from wrapping `mlx::core::compile`:

> Equivalence tests rerun with `mlx_compile: true`; benchmark speedup on
> a Qwen3 forward pass. **If <20% win, drop.**
>
> — `PLAN.md:186-189`

Rather than pay the full Backend/Compiler integration cost to find out,
we answer the question in pure C++ against the vendored MLX 0.25.1 on the
same Apple Silicon target Emily runs on. If compile doesn't help a
transformer block in raw C++, it can't help under BEAM.

## Setup

- Binary: `bench/native/compile_microbench.cpp`
- Harness: `mix bench.native` (Mix task that invokes the `bench-native`
  target in the root `Makefile` with the same env elixir_make sets)
- MLX: 0.25.1 (cocoa-xu prebuilt)
- Host: M-series Mac (Metal GPU)
- Each benchmark runs 50-iteration warmup + 1000 measured iterations
  (500 for seq=512), reporting min/median/p95 wall-time per iteration.
- Both variants call `mx::eval(out); mx::synchronize()` at the end of
  every iteration so compile vs. uncompiled are compared apples-to-apples.

## Results

### Sanity: 8-op elementwise chain (1M elements)

Validates the harness: a pure elementwise workload is exactly what
`mx::compile` is designed to fuse.

| Device | Variant    | min (ms) | median (ms) | p95 (ms) |
|--------|------------|---------:|------------:|---------:|
| GPU    | uncompiled |    1.656 |       1.746 |    2.016 |
| GPU    | compiled   |    0.552 |       0.628 |    0.727 |
| CPU    | uncompiled |    1.423 |       1.463 |    1.625 |
| CPU    | compiled   |    0.986 |       0.997 |    1.036 |

**GPU speedup: 2.78× median** (fusion collapses 8 kernel launches into 1).
**CPU speedup: 1.47× median**. Harness verified.

### Transformer block — Qwen3-0.6B-shaped (seq=128)

RMSNorm → Q/K/V proj → SDPA (matmul, scale, softmax, matmul) → output
proj → residual → RMSNorm → SwiGLU FFN → residual. hidden=1024,
heads=16, head_dim=64, intermediate=2816, batch=1.

| Device | Variant    | min (ms) | median (ms) | p95 (ms) |
|--------|------------|---------:|------------:|---------:|
| GPU    | uncompiled |    2.588 |       3.072 |    3.964 |
| GPU    | compiled   |    2.624 |       2.943 |    3.536 |
| CPU    | uncompiled |    6.835 |       7.151 |    8.109 |
| CPU    | compiled   |    7.792 |       8.082 |    8.810 |

**GPU speedup: 1.04× median — FAILS 1.20× gate.**
**CPU speedup: 0.88× median — compile is slower on CPU.**

### Transformer block — longer seq (seq=512)

Tests whether scaling attention (which grows O(seq²)) shifts the fusion
ratio. It does not.

| Device | Variant    | min (ms) | median (ms) | p95 (ms) |
|--------|------------|---------:|------------:|---------:|
| GPU    | uncompiled |   10.804 |      11.463 |   12.068 |
| GPU    | compiled   |   10.206 |      10.758 |   11.240 |
| CPU    | uncompiled |   21.067 |      21.744 |   22.524 |
| CPU    | compiled   |   25.750 |      26.544 |   27.340 |

**GPU speedup: 1.07× median — FAILS 1.20× gate.**
**CPU speedup: 0.82× median.**

## Interpretation

1. The harness is correct: a pure-elementwise sanity workload yields
   the expected 2-3× compile win.
2. A transformer block is matmul-dominated. MLX's `mx::compile` fuses
   elementwise chains but does **not** fuse matmul kernels with their
   surrounding elementwise ops. The fusion surface (RMSNorm chains,
   softmax neighbourhood, SwiGLU's silu×up) is a small fraction of
   block runtime, bounding the whole-block speedup to single-digit
   percent on GPU.
3. On CPU, compile is a **regression**. Tape-replay overhead exceeds
   fusion gains for workloads that aren't Metal-kernel-launch-bound.
4. Scaling sequence length (128 → 512) does not materially change the
   ratio. This isn't a "small workload" problem; it's a workload-shape
   problem.

## Decision

**Drop M6.** The Phase-1 gate is not met and the measurement explains
why in a way that Phase-2/3 BEAM integration cannot change: the
BEAM-integrated compile path cannot outperform its C++ ceiling, and
that ceiling is 1.04–1.10× on the target workload (transformer
inference).

The microbench source and harness remain in `bench/native/` so this
result can be re-measured against future MLX releases — if MLX adds
matmul-adjacent fusion (e.g. bias-fused matmul or attention fusion
outside `fast::scaled_dot_product_attention`), M6 becomes worth
revisiting.

## Reproduce

```bash
mix bench.native                        # default: warmup 50, iters 1000, seq 128
mix bench.native -- --seq 512 --iters 500
mix bench.native -- --warmup 20 --iters 200   # quick smoke run
```
