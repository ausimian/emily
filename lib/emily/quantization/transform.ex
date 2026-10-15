# Compile-time gate: only define the module when :axon is in the dep
# tree. Emily itself still compiles cleanly without it; the name
# `Emily.Quantization.Transform` is simply undefined until Axon is
# present.
if Code.ensure_loaded?(Axon) do
  defmodule Emily.Quantization.Transform do
    @moduledoc """
    Quantize the dense layers of an Axon model, graph and parameters
    together, so a stock Axon or Bumblebee model runs its matmuls
    through MLX's group-wise quantized kernels.

    Every `:dense` node is rewritten to a `:quantized_dense` node that
    calls `Emily.Quantization.Layers.quantized_dense/4`, and every dense
    kernel in the `Axon.ModelState` is replaced with an
    `%Emily.QuantizedWeight{}`. Biases stay as float tensors. The
    approach mirrors `Axon.Quantization`, but targets MLX quantization
    via `Emily.QuantizedWeight`.

    ## Optional dependency

    This module depends on `:axon`, which is declared `optional: true`
    in Emily's `mix.exs`. Consumers who don't pull Axon into their own
    project get a clean build: the module definition is wrapped in
    `Code.ensure_loaded?/1` and elides entirely when Axon is missing.
    To use it, add Axon to your own `deps/0` (Bumblebee already depends
    on it):

        {:axon, "~> 0.8"}

    ## Usage

    `quantize/3` takes the model and its parameters and returns both
    rewritten:

        {:ok, %{model: model, params: params} = model_info} =
          Bumblebee.load_model({:hf, "Qwen/Qwen3-0.6B"})

        {qmodel, qparams} =
          Emily.Quantization.Transform.quantize(model, params, bits: 4, group_size: 64)

        qmodel_info = %{model_info | model: qmodel, params: qparams}

    The quantized weights are evaluated before `quantize/3` returns, so
    they no longer depend on the dense kernels they came from. Once the
    caller drops its reference to the original parameters (`params`
    above), the dense kernels are freed and resident memory falls to
    the quantized size.

    A small model end to end:

        iex> model =
        ...>   Axon.input("features", shape: {nil, 64})
        ...>   |> Axon.dense(32, name: "fc")
        iex> {init_fn, _predict_fn} = Axon.build(model)
        iex> params = init_fn.(Nx.template({1, 64}, :f32), Axon.ModelState.empty())
        iex> {qmodel, qparams} =
        ...>   Emily.Quantization.Transform.quantize(model, params, bits: 4, group_size: 64)
        iex> Axon.properties(qmodel)["fc"]
        :quantized_dense
        iex> %Emily.QuantizedWeight{bits: 4, group_size: 64} = qparams.data["fc"]["kernel"]
        iex> {_init_fn, predict_fn} = Axon.build(qmodel)
        iex> predict_fn.(qparams, Nx.iota({2, 64}, type: :f32)) |> Nx.shape()
        {2, 32}

    ## Options

    `quantize/3`, `quantize_dense_layers/2` and `quantize_model_state/3`
    accept the same options:

      * `:bits` — one of `#{inspect(Emily.Quantization.defn_supported_bits())}`.
        Default `4`.
      * `:group_size` — elements per quantization group. Default `128`.
      * `:transpose` — default `true`. Groups run along the reduction
        axis: each `[in, out]` kernel is transposed to `[out, in]` before
        quantizing, which is the accuracy convention AWQ uses for LLM
        weights. Set `false` to keep the kernel's stored layout.
      * `:mode` — default `"affine"`. One of `"mxfp4"`, `"mxfp8"`, or
        `"nvfp4"` selects MLX's microscaled modes. `mxfp4` and `nvfp4`
        use FP4-E2M1 lanes; `mxfp8` uses FP8-E4M3 lanes. `mxfp4` and
        `mxfp8` use FP8-E8M0 per-group scales (`group_size = 32`);
        `nvfp4` uses FP8-E4M3 per-group scales (`group_size = 16`,
        finer-grained per the NVIDIA convention).
      * `:except` — list of layer-name substrings. Dense layers whose
        name contains any of them stay as float `:dense` layers. Default
        `[]`. When calling `quantize_dense_layers/2` and
        `quantize_model_state/3` separately, pass the same list to both.

    ## Why not a defn-native `from_dense`

    Axon's param initializers run inside an `Nx.Defn.jit` trace, but
    `Emily.QuantizedWeight.from_dense/2` calls a NIF. So the rewriter
    declares kernels as float templates (matching `Axon.dense`) and
    `quantize_model_state/3` swaps them for `%QuantizedWeight{}` *after*
    init — the same pattern `Axon.Quantization.quantize_model_state/2`
    uses.
    """

    alias Emily.Quantization.Layers
    alias Emily.QuantizedWeight

    @default_opts [bits: 4, group_size: 128, transpose: true, mode: "affine", except: []]
    @supported_modes ~w[affine mxfp4 mxfp8 nvfp4]

    @typedoc "An `Axon.ModelState` struct (Axon defines no type for it)."
    @type model_state :: %Axon.ModelState{}

    # Layers quantized per blocking evaluation in `quantize_model_state/3`.
    # Each batch is evaluated before the next is built, which bounds how
    # much quantization work (and transient memory) is in flight at once.
    @eval_batch_size 8

    @doc """
    Rewrite the model graph and quantize its model state in one call.

    Equivalent to calling `quantize_dense_layers/2` and
    `quantize_model_state/3` with the same options. See the moduledoc
    for the options.
    """
    @spec quantize(Axon.t(), model_state(), keyword()) ::
            {Axon.t(), model_state()}
    def quantize(%Axon{} = model, %Axon.ModelState{} = model_state, opts \\ []) do
      opts = validate_opts!(opts)

      qmodel = quantize_dense_layers(model, opts)
      qstate = quantize_model_state(model, model_state, opts)
      {qmodel, qstate}
    end

    @doc """
    Rewrite every `:dense` node in `model` into a `:quantized_dense` node
    that dispatches through `Emily.Quantization.Layers.quantized_dense/4`.

    The replacement layer's kernel parameter is declared with an
    `[in, out]` float template, identical to `Axon.dense`'s kernel, so
    the `init_fn` from `Axon.build/2` on the rewritten model still works
    and returns a valid (if unquantized) placeholder state. Before
    calling `predict_fn`, either swap that state's kernels for
    `%QuantizedWeight{}` with `quantize_model_state/3`, or supply a
    state whose kernels are already quantized.

    Layer names are preserved, including generated ones such as
    `"dense_0"`, so an existing float `Axon.ModelState`'s keys still
    line up after quantization.

        iex> model =
        ...>   Axon.input("features", shape: {nil, 64})
        ...>   |> Axon.dense(32, name: "fc")
        ...>   |> Axon.relu(name: "act")
        iex> qmodel = Emily.Quantization.Transform.quantize_dense_layers(model)
        iex> props = Axon.properties(qmodel)
        iex> {props["fc"], props["act"]}
        {:quantized_dense, :relu}

    """
    @spec quantize_dense_layers(Axon.t(), keyword()) :: Axon.t()
    def quantize_dense_layers(%Axon{} = model, opts \\ []) do
      opts = validate_opts!(opts)
      dense_layers = dense_layers(model, opts[:except])

      model
      |> pin_dense_names()
      |> Axon.rewrite_nodes(fn
        %Axon.Node{op: :dense, id: id, meta: meta} when is_map_key(dense_layers, id) ->
          fn [%Axon{} = x], _output ->
            quantized_dense(x, meta[:units], use_bias: meta[:use_bias], name: dense_layers[id])
          end

        _ ->
          :skip
      end)
    end

    @doc """
    Walk `model_state` and replace each `:dense` layer's `kernel` tensor
    with an `Emily.QuantizedWeight` container. `bias` is left as a float
    tensor.

    `model` must be the *original* dense model (pre-rewrite):
    `Axon.properties/1` is consulted to find the dense layer names.
    Mirrors `Axon.Quantization.quantize_model_state/2`.

    Handles both `:transpose` conventions:

      * `transpose: true` (default) — transposes each kernel from
        `[in, out]` to `[out, in]` before quantizing, so groups align
        with the reduction axis (the AWQ-accuracy convention for LLM
        weights).
      * `transpose: false` — quantizes the kernel in place, `[in, out]`
        to packed `[in, out_packed]`. Groups run along `out`.

    The quantized weights are evaluated before this function returns,
    a few layers at a time. MLX is lazy, so an unevaluated quantized
    weight keeps its dense source kernel alive as a pending input. With
    the weights evaluated, the dense kernels are freed as soon as the
    caller drops the original `model_state`.

        iex> model =
        ...>   Axon.input("features", shape: {nil, 64})
        ...>   |> Axon.dense(128, name: "fc")
        iex> {init_fn, _predict_fn} = Axon.build(model)
        iex> params = init_fn.(Nx.template({1, 64}, :f32), Axon.ModelState.empty())
        iex> qparams =
        ...>   Emily.Quantization.Transform.quantize_model_state(model, params,
        ...>     group_size: 64,
        ...>     transpose: false
        ...>   )
        iex> qw = qparams.data["fc"]["kernel"]
        iex> {qw.transpose, Nx.shape(qw.value)}
        {false, {64, 16}}
        iex> Nx.shape(qparams.data["fc"]["bias"])
        {128}

    """
    @spec quantize_model_state(Axon.t(), model_state(), keyword()) ::
            model_state()
    def quantize_model_state(%Axon{} = model, %Axon.ModelState{} = state, opts \\ []) do
      opts = validate_opts!(opts)
      quantize_opts = Keyword.take(opts, [:bits, :group_size, :transpose, :mode])

      model
      |> dense_layers(opts[:except])
      |> Map.values()
      |> Enum.chunk_every(@eval_batch_size)
      |> Enum.reduce(state, &quantize_batch(&2, &1, quantize_opts))
    end

    # -- internal ----------------------------------------------------

    # `%{node_id => layer_name}` for every `:dense` node not matched by
    # `except`. Both halves of the transform select layers through this,
    # so the graph rewrite and the state swap always agree.
    defp dense_layers(%Axon{nodes: nodes} = model, except) do
      for {id, name} <- layer_names(model),
          match?(%Axon.Node{op: :dense}, nodes[id]),
          not skip_name?(name, except),
          into: %{},
          do: {id, name}
    end

    # Give every `:dense` node its resolved name as a fixed string. Axon
    # numbers unnamed layers per op, so once some dense layers become
    # `:quantized_dense`, an unnamed one left as `:dense` (by `:except`)
    # would be renumbered and lose its parameters.
    defp pin_dense_names(%Axon{nodes: nodes} = model) do
      names = layer_names(model)

      nodes =
        Map.new(nodes, fn
          {id, %Axon.Node{op: :dense} = node} when is_map_key(names, id) ->
            name = names[id]
            {id, %{node | name: fn _op_name, _op_counts -> name end}}

          entry ->
            entry
        end)

      %{model | nodes: nodes}
    end

    # Resolve each node's layer name the way `Axon.properties/1` does:
    # parents first, with each `name_fn` seeing the op counts so far.
    # The rewrite passes these names on as strings, so an unnamed
    # `dense_0` stays `dense_0` rather than becoming `quantized_dense_0`
    # and losing its parameters.
    defp layer_names(%Axon{output: id, nodes: nodes}) do
      {names, _op_counts} = resolve_names(id, nodes, {%{}, %{}})
      names
    end

    defp resolve_names(id, _nodes, {names, _op_counts} = acc) when is_map_key(names, id),
      do: acc

    defp resolve_names(id, nodes, acc) do
      %Axon.Node{parent: parents, name: name_fn, op_name: op_name} = nodes[id]
      {names, op_counts} = Enum.reduce(parents, acc, &resolve_names(&1, nodes, &2))
      name = name_fn.(op_name, op_counts)
      {Map.put(names, id, name), Map.update(op_counts, op_name, 1, &(&1 + 1))}
    end

    # Build the quantized-dense layer's Axon sub-graph. Declares kernel
    # with a `[in, out]` float template so init_fn succeeds; the predict
    # path pattern-matches on %QuantizedWeight{} and therefore assumes
    # the state has been quantized via `quantize_model_state/3` (or a
    # pre-quantized checkpoint).
    defp quantized_dense(x, units, opts) do
      opts =
        Keyword.validate!(opts, [
          :name,
          use_bias: true,
          kernel_initializer: :glorot_uniform,
          bias_initializer: :zeros
        ])

      # Axon 0.8 dropped `Axon.Shape.dense_kernel/dense_bias`; inline
      # the shape derivation — input is `{batch, …, in_features}`,
      # kernel is `{in_features, units}`, bias is `{units}`.
      kernel_shape = fn input_shape ->
        in_features = elem(input_shape, tuple_size(input_shape) - 1)
        {in_features, units}
      end

      bias_shape = fn _input_shape -> {units} end

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

    # Quantize one batch of layers' kernels, then evaluate them before
    # returning so the batch holds no pending reference to its sources.
    defp quantize_batch(state, layer_names, opts) do
      state =
        Enum.reduce(layer_names, state, fn layer_name, acc ->
          update_in(acc, [Access.key!(:data), layer_name, "kernel"], &quantize_kernel(&1, opts))
        end)

      layer_names
      |> Enum.map(&state.data[&1]["kernel"])
      |> eval_quantized!()

      state
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
        transpose: transpose,
        mode: opts[:mode]
      )
    end

    # `from_dense/2` returns lazy MLX arrays whose pending graph holds
    # the dense source. Schedule the whole batch, then block on each
    # array so the batch has finished before the next one is built.
    defp eval_quantized!(quantized_weights) do
      refs =
        for %QuantizedWeight{} = qw <- quantized_weights,
            %Nx.Tensor{data: %Emily.Backend{ref: ref}} <- [qw.value, qw.scales, qw.biases],
            do: ref

      :ok = Emily.async_eval(refs)
      Enum.each(refs, fn ref -> :ok = Emily.eval(ref) end)
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
                "#{inspect(supported)}. Got: #{inspect(opts[:bits])}"
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

      unless opts[:mode] in @supported_modes do
        raise ArgumentError,
              "Emily.Quantization.Transform: :mode must be one of " <>
                "#{inspect(@supported_modes)}. Got: #{inspect(opts[:mode])}"
      end

      opts
    end
  end
end
