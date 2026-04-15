defmodule Emily.Quantization.Transform do
  @moduledoc """
  Rewrite an Axon graph + `Axon.ModelState` so every dense layer runs
  through `Emily.Quantization.Layers.quantized_dense/4`. Mirrors
  `Axon.Quantization` but targets MLX affine quantization via
  `Emily.QuantizedWeight`.

  Options (see `quantize/3`, `quantize_dense_layers/2`,
  `quantize_model_state/3`):

    * `:bits` — one of `#{inspect([2, 4, 8])}` (defn-native path).
      Default `4`.
    * `:group_size` — elements per quantization group. Default `128`.
    * `:transpose` — default `true`. Groups run along the reduction
      axis (transposes `[in, out]` → `[out, in]` before quantizing —
      the AWQ-accuracy convention for LLMs). Set `false` to keep the
      kernel's stored layout.
    * `:except` — list of layer-name substrings to skip. Only honored
      by `quantize_model_state/3`; pass the same list to
      `quantize_dense_layers/2` for API symmetry.

  Lives under `test/support/` because Axon is an `only: :test` dep;
  graduates to `lib/` alongside a production AWQ-loading path.

  ## Why not a defn-native `from_dense`

  Axon's param initializers run inside an `Nx.Defn.jit` trace, but
  `QuantizedWeight.from_dense/2` calls a NIF. So the rewriter declares
  kernels as float templates (matching `Axon.dense`) and
  `quantize_model_state/3` swaps them for `%QuantizedWeight{}` *after*
  init — the same pattern `Axon.Quantization.quantize_model_state/2`
  uses.
  """

  alias Emily.Quantization.Layers
  alias Emily.QuantizedWeight

  @default_opts [bits: 4, group_size: 128, transpose: true, except: []]

  @doc """
  Rewrite model graph + model state in one call. See moduledoc for
  details.
  """
  @spec quantize(Axon.t(), Axon.ModelState.t(), keyword()) ::
          {Axon.t(), Axon.ModelState.t()}
  def quantize(%Axon{} = model, %Axon.ModelState{} = model_state, opts \\ []) do
    opts = validate_opts!(opts)

    qmodel = quantize_dense_layers(model, opts)
    qstate = quantize_model_state(model, model_state, opts)
    {qmodel, qstate}
  end

  @doc """
  Rewrite every `:dense` node in `model` into a `:quantized_dense` node
  that dispatches through `Emily.Quantization.Layers.quantized_dense/4`.

  The replacement layer's kernel parameter is declared with an `[in,
  out]` float template — identical to `Axon.dense`'s kernel — so that
  `Axon.build(qmodel)`'s `init_fn` works and returns a valid (if
  unquantized) placeholder state. Downstream callers must either swap
  that state's kernels with `%QuantizedWeight{}` via
  `quantize_model_state/3` (dense → quantized round-trip) or supply a
  pre-built quantized state (e.g. from the AWQ loader) before calling
  `predict_fn`.

  Layer names are preserved so an existing float `ModelState`'s keys
  still line up after quantization.
  """
  @spec quantize_dense_layers(Axon.t(), keyword()) :: Axon.t()
  def quantize_dense_layers(%Axon{} = model, opts \\ []) do
    # Options accepted+validated for API-symmetry with
    # `quantize_model_state/3`, but nothing here is opt-sensitive: every
    # `:dense` becomes `:quantized_dense`. `:except` is honored on the
    # state side only (see `quantize_model_state/3`); unmatched layers
    # there keep float kernels and will fail the predict-time pattern
    # match — so pass identical opts to both functions.
    validate_opts!(opts)

    rewriter = fn [%Axon{} = x], _output, name_fn, units, use_bias ->
      quantized_dense(x, units, use_bias: use_bias, name: name_fn)
    end

    Axon.rewrite_nodes(model, fn
      %Axon.Node{op: :dense, meta: meta, name: name_fn} ->
        &rewriter.(&1, &2, name_fn, meta[:units], meta[:use_bias])

      _ ->
        :skip
    end)
  end

  @doc """
  Walk `model_state` and replace each `:dense` layer's `kernel` tensor
  with an `Emily.QuantizedWeight` container. `bias` is left as a float
  tensor.

  `model` must be the *original* dense model (pre-rewrite) —
  `Axon.properties/1` is consulted to find the dense layer names.
  Mirrors `Axon.Quantization.quantize_model_state/2`.

  Handles both `:transpose` conventions:

    * `transpose: true` (default) — transposes each kernel from `[in,
      out]` to `[out, in]` before quantizing, so groups align with the
      reduction axis (the AWQ-accuracy convention for LLM weights).
    * `transpose: false` — quantizes the kernel in-place with `[in,
      out]` → packed `[in, out_packed]`. Groups run along `out`.
  """
  @spec quantize_model_state(Axon.t(), Axon.ModelState.t(), keyword()) ::
          Axon.ModelState.t()
  def quantize_model_state(%Axon{} = model, %Axon.ModelState{} = state, opts \\ []) do
    opts = validate_opts!(opts)
    bits = opts[:bits]
    group_size = opts[:group_size]
    transpose = opts[:transpose]
    except = opts[:except]

    dense_layer_names =
      model
      |> Axon.properties()
      |> Enum.filter(fn {name, op} -> op == :dense and not skip_name?(name, except) end)
      |> Enum.map(fn {name, _} -> name end)

    Enum.reduce(dense_layer_names, state, fn layer_name, acc ->
      update_in(acc, [Access.key!(:data), layer_name, "kernel"], fn kernel ->
        quantize_kernel(kernel, bits: bits, group_size: group_size, transpose: transpose)
      end)
    end)
  end

  # -- internal ----------------------------------------------------

  # Build the quantized-dense layer's Axon sub-graph. Declares kernel
  # with a `[in, out]` float template so init_fn succeeds; the predict
  # path pattern-matches on %QuantizedWeight{} and therefore assumes
  # the state has been quantized via `quantize_model_state/3` (or the
  # AWQ loader).
  defp quantized_dense(x, units, opts) do
    opts =
      Keyword.validate!(opts, [
        :name,
        use_bias: true,
        kernel_initializer: :glorot_uniform,
        bias_initializer: :zeros
      ])

    kernel_shape = &Axon.Shape.dense_kernel(&1, units)
    bias_shape = &Axon.Shape.dense_bias(&1, units)

    kernel = Axon.param("kernel", kernel_shape, initializer: opts[:kernel_initializer])

    {inputs, op} =
      if opts[:use_bias] do
        bias = Axon.param("bias", bias_shape, initializer: opts[:bias_initializer])
        {[x, kernel, bias], &Layers.quantized_dense/4}
      else
        {[x, kernel], &Layers.quantized_dense/3}
      end

    meta = %{units: units, use_bias: opts[:use_bias]}
    Axon.layer(op, inputs, name: opts[:name], meta: meta, op_name: :quantized_dense)
  end

  defp quantize_kernel(kernel, opts) when is_struct(kernel, Nx.Tensor) do
    transpose = opts[:transpose]

    # For transpose=true the stored layout is `[out, in]` (MLX /
    # PyTorch convention, groups along the reduction axis). Axon's
    # kernel is `[in, out]`, so transpose first before calling
    # from_dense.
    source =
      if transpose do
        Nx.transpose(kernel)
      else
        kernel
      end

    QuantizedWeight.from_dense(source,
      group_size: opts[:group_size],
      bits: opts[:bits],
      transpose: transpose
    )
  end

  defp skip_name?(_name, []), do: false

  defp skip_name?(name, patterns) when is_binary(name) do
    Enum.any?(patterns, fn p -> String.contains?(name, p) end)
  end

  defp validate_opts!(opts) do
    opts = Keyword.validate!(opts, @default_opts)

    supported = Emily.Quantization.defn_supported_bits()

    unless opts[:bits] in supported do
      raise ArgumentError,
            "Emily.Quantization.Transform: :bits must be one of " <>
              "#{inspect(supported)} (defn-native dequantize supports " <>
              "integral lanes-per-u32). Got: #{inspect(opts[:bits])}"
    end

    unless is_integer(opts[:group_size]) and opts[:group_size] > 0 do
      raise ArgumentError,
            "Emily.Quantization.Transform: :group_size must be a positive integer, " <>
              "got: #{inspect(opts[:group_size])}"
    end

    unless is_boolean(opts[:transpose]) do
      raise ArgumentError,
            "Emily.Quantization.Transform: :transpose must be a boolean, " <>
              "got: #{inspect(opts[:transpose])}"
    end

    opts
  end
end
