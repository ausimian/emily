defmodule Emily do
  @moduledoc """
  Elixir bindings and Nx backend for Apple's MLX.

  Most users do not call this module directly. `Emily.Backend` plugs
  Emily into `Nx` so every `Nx.*` op runs on MLX, and `Emily.Compiler`
  plugs it into `defn` for `Nx.Serving` / Bumblebee. The functions
  below exist for diagnostics and low-level binary round-tripping:
  build an MLX tensor from a raw buffer, read it back, inspect
  shape / dtype, and force a flush of the lazy graph.

  ## Public API

    * `from_binary/3` — BEAM binary → MLX tensor reference.
    * `to_binary/1` — MLX tensor reference → BEAM binary. Triggers
      evaluation and emits a `[:emily, :to_binary, *]` telemetry span.
    * `shape/1`, `dtype/1` — metadata without triggering evaluation.
    * `eval/1` — force the lazy graph to materialise (useful for
      benchmarking and telemetry). Emits `[:emily, :eval, *]`.

  The everyday integration points are `Emily.Backend`,
  `Emily.Compiler`, and `Emily.Stream`; see each module's moduledoc
  for details.

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
  @type dtype :: Nx.Type.t()

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
  is ready. The returned binary is a BEAM resource binary aliasing
  the MLX buffer directly — no extra memcpy. It stays alive as long
  as any process holds a reference. The same path runs under
  `Nx.to_binary/1` on an `Emily.Backend` tensor.

  The binary is safe to read as immutable: while the binary is
  reachable from BEAM, the underlying buffer's refcount is held
  above 1, which disqualifies it from MLX's buffer-donation
  optimisation (the only path by which MLX writes into an existing
  buffer). MLX is otherwise functional — ops produce fresh arrays.

  > #### Memory accounting {: .warning}
  >
  > The BEAM binary heap accounts the resource binary at the size of
  > the ProcBin (~64 bytes), not the size of the aliased MLX buffer.
  > In a tight loop calling `to_binary/1` on large tensors, the BEAM
  > may not GC aggressively enough to release the underlying MLX
  > storage in a timely fashion — active MLX memory can grow well
  > beyond what the binary heap suggests.
  >
  > Mitigations:
  >
  >   * Force a collection periodically with `:erlang.garbage_collect/0`.
  >   * Let the binary escape to a short-lived process where GC runs
  >     naturally on exit.
  >   * Materialise to a fresh BEAM-owned binary with
  >     `:binary.copy/1` if you only need the bytes briefly.

  ## Examples

      iex> t = Emily.from_binary(<<1.0::float-32-native>>, [1], {:f, 32})
      iex> Emily.to_binary(t)
      <<1.0::float-32-native>>

  """
  @spec to_binary(t()) :: binary()
  def to_binary(tensor) do
    metadata = %{shape: Native.shape(tensor), dtype: Native.dtype(tensor)}

    :telemetry.span([:emily, :to_binary], metadata, fn ->
      bytes = Native.to_binary(Emily.MlxStream.default_worker(), tensor)
      {bytes, Map.put(metadata, :byte_size, byte_size(bytes))}
    end)
  end

  @doc """
  Return the tensor's shape as a list of non-negative ints.

  Reads the lazy-graph metadata; does not trigger evaluation.

  ## Examples

      iex> t = Emily.from_binary(<<0::32, 0::32, 0::32, 0::32>>, [2, 2], {:f, 32})
      iex> Emily.shape(t)
      [2, 2]

  """
  @spec shape(t()) :: [non_neg_integer()]
  defdelegate shape(tensor), to: Native

  @doc """
  Return the tensor's dtype as an `{atom, bits}` tuple.

  Reads the lazy-graph metadata; does not trigger evaluation.

  ## Examples

      iex> t = Emily.from_binary(<<1.0::float-32-native>>, [1], {:f, 32})
      iex> Emily.dtype(t)
      {:f, 32}

  """
  @spec dtype(t()) :: dtype()
  defdelegate dtype(tensor), to: Native

  @doc """
  Force evaluation of the lazy graph rooted at `tensor`.

  Useful for benchmarking or flushing pending work before observing
  side effects. `to_binary/1` implicitly evaluates. Emits a
  `[:emily, :eval, :start | :stop | :exception]` telemetry span.
  """
  @spec eval(t()) :: :ok
  def eval(tensor) do
    :telemetry.span([:emily, :eval], %{}, fn ->
      {Native.eval(Emily.MlxStream.default_worker(), tensor), %{}}
    end)
  end
end
