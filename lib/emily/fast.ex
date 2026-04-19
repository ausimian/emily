defmodule Emily.Fast do
  @moduledoc """
  Fused transformer kernels as `defn`-callable helpers.

  Each function here emits an optional-expression node whose op name
  matches a custom callback on `Emily.Backend`. Under Emily the
  Defn evaluator dispatches directly to the MLX `mx::fast::*`
  kernel; under any other backend the defn-composed fallback runs
  and produces a mathematically equivalent result. That means
  Bumblebee models rewritten to use these helpers (via the test-only
  `Emily.Bumblebee.FastKernels` shim) keep running on
  `Nx.BinaryBackend` / EXLA for conformance — just without the
  fusion speedup.

  ## Hook mechanism

  Each helper wraps `Nx.Defn.Expr.optional(name, in_args, fallback)`,
  which creates an `:optional` Expr node. At eval time the Nx
  evaluator looks for `function_exported?(backend, name,
  length(in_args) + 1)` on the active backend and, if present, calls
  `backend.name(out, args...)` directly. Otherwise the fallback defn
  runs. This is Nx's extension point for vendor-fused kernels (the
  same pattern EXLA uses for its native ops).

  ## Tensor vs option arguments

  The optional-expression contract splits `in_args` at the first
  list: every leading non-list argument is treated as a tensor param;
  the list (typically a keyword list) is passed through as opts. Every
  `Emily.Fast.*` function's final argument is a keyword list of
  scalars (dims, epsilons, flags), and all tensor inputs come before
  it.

  ## Covered kernels

    * `rms_norm/3` — `mx::fast::rms_norm`.
    * `layer_norm/4` — `mx::fast::layer_norm`.
    * `rope/3` — `mx::fast::rope` with the standard geometric-progression
      theta schedule.
    * `rope_with_freqs/4` — `mx::fast::rope` with a precomputed
      inverse-frequency table (for Llama-3 / LongRoPE / linear /
      dynamic scaling).
    * `scaled_dot_product_attention/4` — `mx::fast::sdpa`, without
      mask or with causal mask.
    * `scaled_dot_product_attention_with_mask/5` — the same with an
      additive bias tensor.

  ## Usage

  Call these from inside a `defn` or `Nx.Defn.jit`-traced function,
  alongside regular `Nx` ops:

      defn block(x, w, b) do
        x
        |> Emily.Fast.layer_norm(w, b, eps: 1.0e-5)
        |> Nx.multiply(0.5)
      end

  Under `Emily.Compiler` the `layer_norm` node dispatches to
  `mx::fast::layer_norm`; under `Nx.Defn.Evaluator` + any other
  backend it runs the composed fallback.
  """

  alias Nx.Defn.Expr

  # =================================================================
  # RMSNorm
  # =================================================================

  @doc """
  Fused RMSNorm: `x * rsqrt(mean(x², axis=-1) + eps) * weight`.

  Normalises the last axis of `x`. `weight` must have shape
  `{axis_size(x, -1)}` and broadcasts across the preceding dims.

  `opts`:

    * `:eps` — small constant added inside the rsqrt. Default `1.0e-6`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], backend: Emily.Backend)
      iex> w = Nx.tensor([1.0, 1.0, 1.0, 1.0], backend: Emily.Backend)
      iex> y = Nx.Defn.jit_apply(
      ...>   fn x, w -> Emily.Fast.rms_norm(x, w, eps: 1.0e-5) end,
      ...>   [x, w]
      ...> )
      iex> Nx.shape(y)
      {1, 4}

  """
  @spec rms_norm(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def rms_norm(x, weight, opts \\ []) do
    opts = Keyword.validate!(opts, eps: 1.0e-6)
    Expr.optional(:fast_rms_norm, [x, weight, opts], &rms_norm_fallback/3)
  end

  defp rms_norm_fallback(x, weight, opts) do
    eps = opts[:eps]
    # Match MLX's `upcast: :normalization` recipe — compute variance
    # in f32 even if the payload is f16/bf16, then cast back. This is
    # what Bumblebee's `rms_norm_impl_upcast_normalization` does too.
    orig_type = Nx.type(x)
    x_f32 = Nx.as_type(x, :f32)

    variance = Nx.mean(Nx.pow(x_f32, 2), axes: [-1], keep_axes: true)
    normalized = Nx.multiply(x_f32, Nx.rsqrt(Nx.add(variance, eps)))

    normalized
    |> Nx.as_type(orig_type)
    |> Nx.multiply(weight)
  end

  # =================================================================
  # LayerNorm
  # =================================================================

  @doc """
  Fused LayerNorm: Welford-style mean+variance of the last axis, then
  affine `(x - mean) / sqrt(var + eps) * weight + bias`.

  `weight` and `bias` must both have shape `{axis_size(x, -1)}`.

  `opts`:

    * `:eps` — small constant added inside the sqrt. Default `1.0e-5`.

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], backend: Emily.Backend)
      iex> w = Nx.tensor([1.0, 1.0, 1.0, 1.0], backend: Emily.Backend)
      iex> b = Nx.tensor([0.0, 0.0, 0.0, 0.0], backend: Emily.Backend)
      iex> y = Nx.Defn.jit_apply(
      ...>   fn x, w, b -> Emily.Fast.layer_norm(x, w, b, eps: 1.0e-5) end,
      ...>   [x, w, b]
      ...> )
      iex> Nx.shape(y)
      {1, 4}

  """
  @spec layer_norm(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def layer_norm(x, weight, bias, opts \\ []) do
    opts = Keyword.validate!(opts, eps: 1.0e-5)
    Expr.optional(:fast_layer_norm, [x, weight, bias, opts], &layer_norm_fallback/4)
  end

  defp layer_norm_fallback(x, weight, bias, opts) do
    eps = opts[:eps]
    orig_type = Nx.type(x)
    x_f32 = Nx.as_type(x, :f32)

    mean = Nx.mean(x_f32, axes: [-1], keep_axes: true)
    centered = Nx.subtract(x_f32, mean)
    variance = Nx.mean(Nx.pow(centered, 2), axes: [-1], keep_axes: true)
    normalized = Nx.multiply(centered, Nx.rsqrt(Nx.add(variance, eps)))

    normalized
    |> Nx.as_type(orig_type)
    |> Nx.multiply(weight)
    |> Nx.add(bias)
  end

  # =================================================================
  # Rotary position embedding
  # =================================================================

  @doc """
  Fused RoPE with the standard geometric-progression theta schedule.

  Rotates the trailing `dims` axes of `x` (typically `head_dim`) in
  position-indexed planes. `offset` is a scalar integer tensor
  (usually `Nx.tensor(0)` for prompt-processing, or the KV-cache
  length for incremental decode).

  `opts`:

    * `:dims` — number of trailing axes to rotate. Required.
    * `:traditional` — if `true`, use the paired-interleave layout
      (MLX / Meta convention). If `false`, split-half layout
      (HuggingFace convention). Default `false`.
    * `:base` — theta base. Default `10_000.0`.
    * `:scale` — position scale multiplier. Default `1.0`.

  For scaled variants (Llama-3, LongRoPE, linear, dynamic) use
  `rope_with_freqs/4` with a precomputed inverse-frequency table.

  ## Examples

      iex> x = Nx.iota({1, 1, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> offset = Nx.tensor(0, backend: Emily.Backend)
      iex> y = Nx.Defn.jit_apply(
      ...>   fn x, o -> Emily.Fast.rope(x, o, dims: 8) end,
      ...>   [x, offset]
      ...> )
      iex> Nx.shape(y)
      {1, 1, 4, 8}

  """
  @spec rope(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def rope(x, offset, opts) do
    opts = Keyword.validate!(opts, [:dims, traditional: false, base: 10_000.0, scale: 1.0])
    Expr.optional(:fast_rope, [x, offset, opts], &rope_fallback/3)
  end

  defp rope_fallback(x, offset, opts) do
    dims = opts[:dims]
    base = opts[:base]
    scale = opts[:scale]
    traditional = opts[:traditional]

    half = div(dims, 2)
    range = Nx.iota({half}, type: :f32) |> Nx.multiply(2) |> Nx.divide(dims)
    inv_freq = Nx.pow(base, range) |> then(&Nx.divide(1.0, &1))

    rope_common(x, offset, inv_freq, dims, traditional, scale)
  end

  @doc """
  RoPE with a precomputed inverse-frequency table.

  Use this overload when the model applies a non-standard scaling
  strategy to the base frequencies (e.g. Llama-3, LongRoPE, linear,
  dynamic). `freqs` must be a 1-D `:f32` tensor of length `dims / 2`.

  `opts`:

    * `:dims` — number of trailing axes to rotate. Required.
    * `:traditional` — see `rope/3`. Default `false`.
    * `:scale` — position scale multiplier. Default `1.0`.

  ## Examples

      iex> x = Nx.iota({1, 1, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> offset = Nx.tensor(0, backend: Emily.Backend)
      iex> freqs = Nx.tensor([1.0, 0.1, 0.01, 0.001], backend: Emily.Backend)
      iex> y = Nx.Defn.jit_apply(
      ...>   fn x, o, f -> Emily.Fast.rope_with_freqs(x, o, f, dims: 8) end,
      ...>   [x, offset, freqs]
      ...> )
      iex> Nx.shape(y)
      {1, 1, 4, 8}

  """
  @spec rope_with_freqs(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def rope_with_freqs(x, offset, freqs, opts) do
    opts = Keyword.validate!(opts, [:dims, traditional: false, scale: 1.0])
    Expr.optional(:fast_rope_with_freqs, [x, offset, freqs, opts], &rope_freqs_fallback/4)
  end

  defp rope_freqs_fallback(x, offset, freqs, opts) do
    dims = opts[:dims]
    traditional = opts[:traditional]
    scale = opts[:scale]
    rope_common(x, offset, freqs, dims, traditional, scale)
  end

  # Shared RoPE body. Expected layout on x: `{..., seq, dims}`.
  defp rope_common(x, offset, inv_freq, dims, traditional, scale) do
    orig_type = Nx.type(x)
    x_f32 = Nx.as_type(x, :f32)

    shape = Nx.shape(x_f32)
    rank = tuple_size(shape)
    seq_len = elem(shape, rank - 2)

    positions =
      Nx.iota({seq_len}, type: :f32)
      |> Nx.multiply(scale)
      |> Nx.add(Nx.as_type(offset, :f32))

    angle = Nx.outer(positions, inv_freq)
    cos = Nx.cos(angle)
    sin = Nx.sin(angle)

    rotated =
      if traditional do
        rope_traditional(x_f32, cos, sin)
      else
        rope_half(x_f32, cos, sin, dims)
      end

    Nx.as_type(rotated, orig_type)
  end

  defp rope_half(x, cos, sin, dims) do
    half = div(dims, 2)
    x1 = Nx.slice_along_axis(x, 0, half, axis: -1)
    x2 = Nx.slice_along_axis(x, half, half, axis: -1)

    cos_b = broadcast_trig(cos, x1)
    sin_b = broadcast_trig(sin, x1)

    out1 = Nx.subtract(Nx.multiply(x1, cos_b), Nx.multiply(x2, sin_b))
    out2 = Nx.add(Nx.multiply(x1, sin_b), Nx.multiply(x2, cos_b))
    Nx.concatenate([out1, out2], axis: -1)
  end

  defp rope_traditional(x, cos, sin) do
    shape = Nx.shape(x)
    rank = tuple_size(shape)
    dims = elem(shape, rank - 1)
    half = div(dims, 2)

    pair_shape = shape |> put_elem(rank - 1, half) |> Tuple.insert_at(rank, 2)
    paired = Nx.reshape(x, pair_shape)

    x_even = Nx.slice_along_axis(paired, 0, 1, axis: -1) |> Nx.squeeze(axes: [-1])
    x_odd = Nx.slice_along_axis(paired, 1, 1, axis: -1) |> Nx.squeeze(axes: [-1])

    cos_b = broadcast_trig(cos, x_even)
    sin_b = broadcast_trig(sin, x_even)

    out_even = Nx.subtract(Nx.multiply(x_even, cos_b), Nx.multiply(x_odd, sin_b))
    out_odd = Nx.add(Nx.multiply(x_even, sin_b), Nx.multiply(x_odd, cos_b))

    stacked = Nx.stack([out_even, out_odd], axis: -1)
    Nx.reshape(stacked, shape)
  end

  # cos/sin arrive as {seq, half}; target is {..., seq, half}.
  defp broadcast_trig(trig, target) do
    target_shape = Nx.shape(target)
    rank = tuple_size(target_shape)
    pad_axes = rank - 2
    trig_shape = List.duplicate(1, pad_axes) ++ Tuple.to_list(Nx.shape(trig))
    Nx.broadcast(Nx.reshape(trig, List.to_tuple(trig_shape)), target_shape)
  end

  # =================================================================
  # Scaled dot-product attention
  # =================================================================

  @doc """
  Fused scaled-dot-product attention without an additive-bias mask.

  Expects `{batch, heads, seq, head_dim}` layout on Q, K, V.

  `opts`:

    * `:scale` — multiplier on QKᵀ before softmax. Default
      `1 / sqrt(head_dim)`.
    * `:causal` — if `true`, apply MLX's built-in upper-triangular
      mask. Default `false`.

  ## Examples

      iex> q = Nx.iota({1, 2, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> k = Nx.iota({1, 2, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> v = Nx.iota({1, 2, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> y = Nx.Defn.jit_apply(
      ...>   fn q, k, v -> Emily.Fast.scaled_dot_product_attention(q, k, v) end,
      ...>   [q, k, v]
      ...> )
      iex> Nx.shape(y)
      {1, 2, 4, 8}

  """
  @spec scaled_dot_product_attention(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def scaled_dot_product_attention(q, k, v, opts \\ []) do
    opts = Keyword.validate!(opts, [:scale, causal: false])
    opts = Keyword.put_new_lazy(opts, :scale, fn -> default_sdpa_scale(q) end)

    Expr.optional(:fast_scaled_dot_product_attention, [q, k, v, opts], &sdpa_fallback/4)
  end

  defp sdpa_fallback(q, k, v, opts) do
    scale = opts[:scale]
    causal = opts[:causal]

    # QKᵀ: contract q's last axis with k's last axis, batch-dot on the
    # (batch, heads) leading dims.
    weights = Nx.dot(q, [-1], [0, 1], k, [-1], [0, 1]) |> Nx.multiply(scale)

    weights =
      if causal do
        q_len = Nx.axis_size(q, -2)
        k_len = Nx.axis_size(k, -2)
        mask = Nx.less_equal(Nx.iota({q_len, 1}), Nx.iota({1, k_len}))

        bias =
          Nx.select(
            mask,
            Nx.tensor(0.0, type: Nx.type(weights)),
            Nx.Constants.min_finite(Nx.type(weights))
          )

        Nx.add(weights, Nx.reshape(bias, {1, 1, q_len, k_len}))
      else
        weights
      end

    probs = softmax_last_axis(weights)
    Nx.dot(probs, [-1], [0, 1], v, [-2], [0, 1])
  end

  @doc """
  SDPA with an additive mask tensor broadcasting across QKᵀ.

  `mask` should match (or broadcast to) shape
  `{batch_or_1, heads_or_1, q_len, k_len}` and is added to QKᵀ *after*
  scaling. Use `Nx.Constants.min_finite/1` on positions to mask out.

  `opts`:

    * `:scale` — see `scaled_dot_product_attention/4`. Default
      `1 / sqrt(head_dim)`.

  ## Examples

      iex> q = Nx.iota({1, 2, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> k = Nx.iota({1, 2, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> v = Nx.iota({1, 2, 4, 8}, backend: Emily.Backend, type: :f32)
      iex> mask = Nx.broadcast(Nx.tensor(0.0), {1, 1, 4, 4}) |> Nx.backend_transfer(Emily.Backend)
      iex> y = Nx.Defn.jit_apply(
      ...>   fn q, k, v, m -> Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, m) end,
      ...>   [q, k, v, mask]
      ...> )
      iex> Nx.shape(y)
      {1, 2, 4, 8}

  """
  @spec scaled_dot_product_attention_with_mask(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def scaled_dot_product_attention_with_mask(q, k, v, mask, opts \\ []) do
    opts = Keyword.validate!(opts, [:scale])
    opts = Keyword.put_new_lazy(opts, :scale, fn -> default_sdpa_scale(q) end)

    Expr.optional(
      :fast_scaled_dot_product_attention_with_mask,
      [q, k, v, mask, opts],
      &sdpa_masked_fallback/5
    )
  end

  defp sdpa_masked_fallback(q, k, v, mask, opts) do
    scale = opts[:scale]

    weights =
      q
      |> Nx.dot([-1], [0, 1], k, [-1], [0, 1])
      |> Nx.multiply(scale)
      |> Nx.add(mask)

    probs = softmax_last_axis(weights)
    Nx.dot(probs, [-1], [0, 1], v, [-2], [0, 1])
  end

  defp default_sdpa_scale(q) do
    head_dim = Nx.axis_size(q, -1)
    1.0 / :math.sqrt(head_dim)
  end

  # Numerically stable softmax along the last axis. Inlined here
  # instead of pulling Axon (a test-only dep) into lib/.
  defp softmax_last_axis(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(x, max))
    sum = Nx.sum(exp, axes: [-1], keep_axes: true)
    Nx.divide(exp, sum)
  end
end
