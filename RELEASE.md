### Changed

- Updated the pinned MLX to 0.32.0. This is a maintenance bump that also
  picks up faster small-batch quantized matvec (`qmv_wide`) — accelerating
  the fused quantized path — and broader fused SDPA coverage (asymmetric
  Q/V head dims), both transparently. No API changes.
- Quantized dense layers now use the fused `mx::quantized_matmul` kernel
  instead of dequantizing the full weight to bf16 and running a dense
  matmul. The packed low-bit weights are streamed directly, so a decode
  step no longer re-dequantizes the entire model on every token. On a
  4-bit Qwen3-0.6B this makes native quantized generation roughly 13×
  faster end-to-end — and quantized inference is now *faster* than dense,
  as it should be, rather than slower.
