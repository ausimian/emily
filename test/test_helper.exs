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
#     mix test --only qwen3_full      # ~1.5 GB checkpoint
#     mix test --only vit_full        # ~330 MB checkpoint
#     mix test --only whisper_full    # ~150 MB checkpoint
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
# `:qwen3_quant_full` is the M10.5 quantized end-to-end conformance —
# runs the full Qwen3-0.6B through Transform.quantize/3 and greedy
# decodes 32 tokens. Same 1.5 GB checkpoint as `:qwen3_full`; opt in
# with:
#
#     mix test --only qwen3_quant_full
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
ExUnit.start(
  max_cases: System.schedulers_online(),
  exclude: [
    :conformance,
    :qwen3_full,
    :qwen3_quant_full,
    :vit_full,
    :whisper_full,
    :training_full,
    :fast_kernels_full
  ]
)
