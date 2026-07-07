### Changed

- Quantized dense layers now use the fused `mx::quantized_matmul` kernel
  instead of dequantizing the full weight to bf16 and running a dense
  matmul. The packed low-bit weights are streamed directly, so a decode
  step no longer re-dequantizes the entire model on every token. On a
  4-bit Qwen3-0.6B this makes native quantized generation roughly 13×
  faster end-to-end — and quantized inference is now *faster* than dense,
  as it should be, rather than slower. Non-Emily backends keep the
  composed dequantize + `Nx.dot` fallback.
