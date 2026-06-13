# MLX is now thread-safe (thread-local CommandEncoder, ml-explore/mlx#3348;
# ThreadLocalStream API, ml-explore/mlx#3405). Parallel module execution
# is safe — the safe_eval mutex is removed and concurrent Metal dispatch
# is handled natively by MLX.
#
# Conformance tests pull tiny-random HuggingFace models at runtime and
# take ~tens of seconds per model on a cold cache — opt-in via
# `mix test --only conformance`. The `:*_full` tags are heavier
# conformance variants that download full-size weight checkpoints, so
# they are excluded even from `--only conformance`; run explicitly:
#
#     mix test --only vit_full           # ~330 MB checkpoint
#     mix test --only whisper_full       # ~150 MB checkpoint
#     mix test --only distilbert_full    # ~250 MB checkpoint
#
# Qwen3 conformance lives outside `mix test` entirely —
# `scripts/qwen3_conformance.exs` is a standalone Mix.install script
# pinning the Bumblebee main ref that carries Qwen3 support. See its
# header for usage.
#
# (Soak tests deliberately stay in the default suite; see
# `test/soak/memory_test.exs` for the rationale.)
#
# `:training_full` is the M9 MNIST convergence canary — downloads
# MNIST (~11 MB) via `scidata`, trains an Axon MLP to >97% test
# accuracy. Excluded by default because it's multi-minute wall time;
# run explicitly:
#
#     mix test --only training_full
#
# `:grad_conformance` is the M13 EXLA gradient oracle suite — compares
# Emily grads against EXLA-produced golden values. Lightweight (no
# network, no downloads), runs in the default suite. Select explicitly:
#
#     mix test --only grad_conformance
#
# `:fast_kernels_full` is the M11 fused-kernel variant of every full
# conformance model (and one tiny-random DistilBERT smoke). Each test
# applies `Emily.Bumblebee.FastKernels.apply/1` to the loaded Axon
# model so that RMSNorm / LayerNorm / RoPE / SDPA dispatch through
# the MLX `mx::fast::*` kernels via `Emily.Fast`. Run explicitly:
#
#     mix test --only fast_kernels_full
#
# `:native` and `:fuse` are the expression-compiler lanes of
# the tiny-random conformance suites: every `mode_test` (see
# `Emily.ConformanceHelper`) re-runs the forward pass under
# `compiler: Emily.Compiler, native: true, native_fallback: :raise`
# (`:native`) and again with `fuse: true` wrapping the replay
# in `mx::compile` (`:fuse`), so the same PyTorch reference
# slice validates the evaluator, the native-compiled, and the fused
# paths. Those tests carry `:conformance` too, so `--only conformance`
# runs all three lanes; select one lane alone with:
#
#     mix test --only native
#     mix test --only fuse
#
# Listed in the default exclude defensively — every such test is already
# `:conformance`-tagged, but this keeps a future `:native`-only or
# `:fuse`-only test out of the default suite.
ExUnit.start(
  max_cases: System.schedulers_online(),
  exclude: [
    :conformance,
    :native,
    :fuse,
    :vit_full,
    :whisper_full,
    :distilbert_full,
    :training_full,
    :fast_kernels_full
  ]
)
