### Added

- `Emily.Quantization.Transform` now ships with the package. Previously
  it was test-only code. `quantize/3` rewrites every `:dense` layer of an
  Axon or Bumblebee model to run through
  `Emily.Quantization.Layers.quantized_dense/4` and replaces its kernel
  in the `Axon.ModelState` with an `Emily.QuantizedWeight`.
  `quantize_dense_layers/2` and `quantize_model_state/3` do each half
  separately. Options are `:bits`, `:group_size`, `:transpose`, `:mode`
  and `:except`, which keeps matching layers as float `:dense` layers.
  The module is defined only when Axon is available. The quantized
  weights are evaluated before `quantize/3` returns, so the dense
  kernels are freed as soon as the caller drops the original model state
  instead of staying resident until the first forward pass. The
  `qwen3_quantized` livebook now calls `quantize/3` instead of defining
  its own transform.

### Fixed

- Under `native: true`, `Nx.cumulative_sum/2`, `cumulative_product/2`,
  `cumulative_max/2` and `cumulative_min/2` on any axis but the last now
  read their own operand. They previously used the function's first
  input instead, which gave a silently wrong result when the shapes
  matched and an MLX `[slice]` error when they did not.
