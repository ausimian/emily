defmodule Emily do
  @moduledoc """
  Elixir bindings and Nx backend for Apple's MLX.

  This module exposes a minimal public surface. Higher-level integration
  (the Nx backend and Defn compiler) lives in `Emily.Backend` and
  `Emily.Compiler` — those are not yet implemented (see `PLAN.md`).

  The M0 surface is intentionally tiny: build a tensor from a binary,
  round-trip it back, and inspect shape/dtype. This is the narrowest
  slice that proves the NIF + MLX linking is healthy.

  ## Debug assertions

  Emily supports two compile-time flags that enable runtime assertions
  on hot paths. Both default to `false` with zero runtime cost — the
  guarded branches are dead-code-eliminated by the Elixir compiler
  when the flag is `false`.

      # config/dev.exs  (opt in only in development / CI)
      import Config
      config :emily,
        debug_bounds_check: true,
        debug_detect_nan_inf: true

    * `:debug_bounds_check` — assert indices are in range for
      `gather`, `take`, `take_along_axis`, `indexed_add`, and
      `indexed_put`. Raises `ArgumentError` on out-of-range or
      negative indices. GPU backends (Emily, EXLA, Torch-CUDA,
      JAX-GPU) don't bounds-check by default — an OOB index gets
      whatever bytes happen to live at that memory address, which
      can silently produce `NaN` scores that propagate through
      softmax. Turning this on in CI catches the bug class at the
      offending op.

    * `:debug_detect_nan_inf` — scan results of `matmul`,
      `fast_rms_norm`, `fast_layer_norm`, and the two
      `fast_scaled_dot_product_attention` variants for NaN/Inf.
      Raises `ArgumentError` on detection. Useful during training
      so numerics failures surface at the op that produced them
      rather than downstream as `loss = NaN`. Standalone softmax
      has no backend callback in Emily (it's Axon-composed from
      `exp` / `sum` / `divide`); only the fused SDPA softmax is
      scanned.

  Each assertion forces a small MLX reduction plus a scalar readback
  on the worker — a sync point that breaks lazy-graph fusion and
  adds noticeable overhead. Leave the flags off in release builds.
  """

  alias Emily.Native

  @typedoc "An opaque reference to an MLX tensor."
  @opaque t :: reference()

  @typedoc "An Nx-compatible dtype, e.g. `{:f, 32}` or `{:s, 64}`."
  @type dtype :: {atom(), non_neg_integer()}

  @doc """
  Build a lazy MLX tensor from a raw binary.

  The binary must contain exactly `product(shape) * byte_size(dtype)`
  bytes in native-endian layout. MLX copies the buffer on construction,
  so the binary need not outlive the call.

  ## Examples

      iex> t = Emily.from_binary(<<1.0::float-32-native>>, [1], {:f, 32})
      iex> Emily.to_binary(t)
      <<1.0::float-32-native>>

  """
  @spec from_binary(binary(), [non_neg_integer()], dtype()) :: t()
  def from_binary(data, shape, dtype) when is_binary(data) and is_list(shape) do
    Native.from_binary(data, shape, dtype)
  end

  @doc """
  Materialize the tensor and return its raw bytes.

  Triggers `eval` on the underlying MLX graph; blocks until the result
  is ready.
  """
  @spec to_binary(t()) :: binary()
  def to_binary(tensor) do
    metadata = %{shape: Native.shape(tensor), dtype: Native.dtype(tensor)}

    :telemetry.span([:emily, :to_binary], metadata, fn ->
      bytes = Native.to_binary(Emily.MlxStream.default_worker(), tensor)
      {bytes, Map.put(metadata, :byte_size, byte_size(bytes))}
    end)
  end

  @doc "Return the tensor's shape as a list of non-negative ints."
  @spec shape(t()) :: [non_neg_integer()]
  defdelegate shape(tensor), to: Native

  @doc "Return the tensor's dtype as an `{atom, bits}` tuple."
  @spec dtype(t()) :: dtype()
  defdelegate dtype(tensor), to: Native

  @doc """
  Force evaluation of the lazy graph rooted at `tensor`.

  Useful for benchmarking or flushing pending work before
  observing side effects. `to_binary/1` implicitly evaluates.
  """
  @spec eval(t()) :: :ok
  def eval(tensor) do
    :telemetry.span([:emily, :eval], %{}, fn ->
      {Native.eval(Emily.MlxStream.default_worker(), tensor), %{}}
    end)
  end
end
