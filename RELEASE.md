Initial release. See the git history for per-milestone detail.

### Added

- **Nx backend.** `Emily.Backend` implements every required
  `Nx.Backend` callback against MLX, with transparent fallback to
  `Nx.BinaryBackend` for ops without a native primitive.
- **Defn compiler.** `Emily.Compiler` runs `defn` / `Nx.Serving` /
  Bumblebee on Emily; pins the result backend and caps partition
  concurrency so `Nx.Serving` stays compatible.
- **Fused transformer kernels.** `Emily.Fast` exposes
  `mx::fast::rms_norm`, `layer_norm`, `rope`, and scaled-dot-product
  attention as defn-callable helpers with composed-defn fallbacks
  for non-Emily backends.
- **Affine group-wise quantization.** `Emily.QuantizedWeight` and
  `Emily.Quantization` wrap MLX `quantize` / `dequantize` /
  `quantized_matmul` for int2 / int4 / int8 inference.
  `Emily.Quantization.dequantize_defn/1` provides a defn-native
  dequantize for use inside Axon forward passes.
- **Mixed-precision training.** `Emily.MixedPrecision` ships the
  bf16 recipe: `cast_params` for the forward pass, f32 master
  weights, dynamic loss scaling with overflow detection.
- **Per-process Metal streams.** `Emily.Stream` lets each BEAM
  process own its own Metal command queue, enabling concurrent
  inference on a shared model.
- **Zero-copy `to_binary`.** `Nx.to_binary/1` on an Emily tensor
  returns a BEAM resource binary aliasing the MLX buffer — no memcpy.
- **Native gradient + training primitives.** `gather`, `scatter`,
  `scatter_add`, `conv`, and the window-reduction family lower
  directly to MLX so `Nx.Defn.grad` and CNN training stay native.
- **Native linalg.** `lu`, `svd`, `qr`, `cholesky`, `eigh`, `solve`,
  and `triangular_solve` dispatch to `mx::linalg::*` instead of
  rounding through `Nx.BinaryBackend`.
- **Telemetry.** `[:emily, :eval, *]`, `[:emily, :to_binary, *]`,
  `[:emily, :fallback, *]`, and `[:emily, :memory, :stats]` span
  events; opt-in one-shot fallback warnings via
  `config :emily, :warn_on_fallback, true`.
- **Compile-time debug flags.** `:debug_bounds_check` and
  `:debug_detect_nan_inf` re-enable runtime assertions on hot paths;
  default off with zero runtime cost.
- **Bumblebee conformance.** End-to-end suites for DistilBERT,
  Qwen3-0.6B (dense and quantized), ViT-base, and Whisper-tiny,
  pinned against HuggingFace reference values.
- **Worker-thread dispatch.** Each MLX stream is owned by a
  dedicated OS thread. NIFs enqueue work on the worker and return
  immediately; the worker posts the result back to the caller via
  `enif_send`, and the public wrapper awaits it with `receive`. No
  BEAM scheduler (regular or dirty) blocks on MLX work, and the
  per-thread Metal `CommandEncoder` state stays consistent regardless
  of how the BEAM migrates Elixir processes between schedulers.
- **Vendored MLX build.** MLX is built from source via cmake from
  `vendor/mlx` (git submodule); no prebuilt download. Build cache
  keyed on the submodule SHA under `~/Library/Caches/emily/`.
- **Documentation.** Per-module HexDocs, four runnable Livebooks
  (`notebooks/distilbert_qa.livemd`,
  `notebooks/qwen3_quantized.livemd`,
  `notebooks/mnist_training.livemd`,
  `notebooks/whisper_transcription.livemd`), and worked Bumblebee
  examples in the conformance suite.
