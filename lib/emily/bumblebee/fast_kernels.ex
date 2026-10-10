# Compile-time gate: only define the module when both :axon and
# :bumblebee are in the dep tree. Emily itself still compiles cleanly
# without them; the name `Emily.Bumblebee.FastKernels` is simply
# undefined until both deps are present.
if Code.ensure_loaded?(Axon) and Code.ensure_loaded?(Bumblebee.Layers) do
  defmodule Emily.Bumblebee.FastKernels do
    @moduledoc """
    Rewrite RMSNorm, LayerNorm, RoPE, and SDPA Axon layers of a
    Bumblebee model so they call `Emily.Fast.*` instead of their stock
    defn implementations. When the rewritten model is then evaluated
    under `Emily.Compiler`, those `Emily.Fast.*` calls dispatch to
    fused MLX kernels via the `:optional`-node mechanism (see
    `Emily.Fast`'s moduledoc). On any other backend the helpers fall
    back to defn composition and produce mathematically equivalent
    results, so applying the shim is safe even if the model is later
    evaluated on `Nx.BinaryBackend` or EXLA.

    ## Optional dependency

    This module depends on `:axon` and `:bumblebee`, which are declared
    as `optional: true` in Emily's `mix.exs`. Consumers who don't pull
    those deps into their own project get a clean build: the whole
    module definition is wrapped in `Code.ensure_loaded?/1` and elides
    entirely when either dep is missing. To use the shim, add both to
    your own `deps/0`:

        {:bumblebee, "~> 0.7"},
        {:axon, "~> 0.8"}

    ## Usage

        {:ok, model_info} = Bumblebee.load_model({:hf, "openai/whisper-tiny"})

        model_info =
          update_in(model_info.model, &Emily.Bumblebee.FastKernels.apply/1)

        # then proceed with Bumblebee.Audio.speech_to_text_whisper/5
        # (or Text.generation/4 / etc.) as usual.

    ## Coverage

      * `:rms_norm` (Bumblebee's `rms_norm/2`).
      * `:layer_norm` (Axon's built-in normalization layer).
      * `Bumblebee.Layers.apply_rotary_embedding/5` — supports the
        default schedule plus the `:linear`, `:dynamic`, `:longrope`,
        `:llama3` scaling strategies, following Bumblebee's own
        schedules. Scaled inverse frequencies are precomputed at
        rewrite time and passed to `mx::fast::rope` via the
        `freqs`-override overload. The fused kernel runs for
        single-token calls (cached decode), with each row's position as
        its offset; multi-token calls such as prefill rotate by the
        explicit `position_ids`, so padded and batched inputs keep
        Bumblebee's per-token positions.
      * `Bumblebee.Layers.attention_output_impl/3` — coalesced with
        its sibling `attention_weights_impl/7` into a single
        `mx::fast::scaled_dot_product_attention` dispatch. Mask
        translation: plain and causal self-attention use MLX's built-in
        modes; anything else (key mask, bias, sliding window, or a
        cache offset) is collapsed into one additive `array` mask.

    ## What's *not* rewritten

      * Norms with `:channel_index` other than `-1` (uncommon outside
        vision-CNN heads — those don't fit the fused kernel's
        last-axis-only contract). The original layer is left in place.
      * Attention layers with `dropout_rate > 0`. Inference path is
        `dropout_rate: 0` everywhere, so this is a no-op in practice;
        training paths continue using composed defn.
    """

    alias Nx.Defn.Expr

    # Bumblebee's RoPE / attention impls are `defnp` (private), so we
    # can't capture them as `&BL.x/n` from outside the module. Instead
    # we identify a captured op by its `{module, function, arity}` via
    # `Function.info/2`.
    @rope_mfa {Bumblebee.Layers, :apply_rotary_embedding, 5}
    @attention_output_mfa {Bumblebee.Layers, :attention_output_impl, 3}

    @doc """
    Apply every available rewrite to `model`.
    """
    @spec apply(Axon.t()) :: Axon.t()
    def apply(%Axon{} = model) do
      model
      |> apply_rms_norm()
      |> apply_layer_norm()
      |> apply_rope()
      |> apply_attention()
    end

    # ----------------------------------------------------------------
    # RMSNorm
    # ----------------------------------------------------------------

    @doc false
    def apply_rms_norm(%Axon{} = model) do
      rewrite_last_axis_norm(model, :rms_norm, &fast_rms_norm_impl/3)
    end

    # Replacement for Bumblebee's `rms_norm_impl_upcast_*`. Same arity
    # so it slots into the existing Axon.layer call.
    @doc false
    def fast_rms_norm_impl(input, weight, opts \\ []) do
      opts =
        Keyword.validate!(opts, shift: 0.0, epsilon: 1.0e-6, channel_index: -1, mode: :inference)

      # Bumblebee multiplies by `(shift + weight)`. For shift==0 (the
      # universal default) we hand `weight` straight to the kernel; for
      # the rare nonzero case we pre-add Elixir-side so the fused
      # kernel still sees a single weight tensor.
      shift = opts[:shift]
      weight = if shift == 0, do: weight, else: Nx.add(weight, shift)

      Emily.Fast.rms_norm(input, weight, eps: opts[:epsilon])
    end

    # ----------------------------------------------------------------
    # LayerNorm
    # ----------------------------------------------------------------

    @doc false
    def apply_layer_norm(%Axon{} = model) do
      rewrite_last_axis_norm(model, :layer_norm, &fast_layer_norm_impl/4)
    end

    @doc false
    def fast_layer_norm_impl(input, gamma, beta, opts \\ []) do
      opts = Keyword.validate!(opts, epsilon: 1.0e-5, channel_index: -1, mode: :inference)
      Emily.Fast.layer_norm(input, gamma, beta, eps: opts[:epsilon])
    end

    # Shared rewriter for `:rms_norm` and `:layer_norm` — both normalise
    # over the last axis and both gate on `channel_index == -1`.
    defp rewrite_last_axis_norm(model, op_name, impl) do
      Axon.map_nodes(model, fn
        %Axon.Node{op_name: ^op_name, opts: layer_opts} = node ->
          if Keyword.get(layer_opts, :channel_index, -1) == -1 do
            %{node | op: impl}
          else
            node
          end

        other ->
          other
      end)
    end

    # ----------------------------------------------------------------
    # RoPE
    # ----------------------------------------------------------------
    #
    # Bumblebee's apply_rotary_embedding/5 takes (query, key,
    # position_ids, attention_mask, opts) and returns
    # {rotated_query, rotated_key} as a tuple. We replace it with a
    # function that has the same shape contract. Scaled schedules get
    # their inverse frequencies precomputed once at rewrite time.
    #
    # `position_ids` can hold any per-token positions: Bumblebee's
    # generation derives them from the attention mask, so padded or
    # batched prompts are neither contiguous nor equal across rows. The
    # fused kernel only takes one starting position per batch entry, so
    # it handles single-token calls (cached decode, the hot path), and
    # longer calls rotate by their explicit positions instead.

    @doc false
    def apply_rope(%Axon{} = model) do
      Axon.map_nodes(model, fn
        %Axon.Node{op: op, opts: opts} = node ->
          if fn_mfa(op) == @rope_mfa do
            # Stash the precomputed freqs in the node's opts so the
            # per-layer-per-token hot path (`fast_rope_impl`) can grab
            # them directly instead of redoing the arithmetic on every
            # call.
            new_opts = Keyword.put(opts, :precomputed_freqs, precompute_freqs(opts))
            %{node | op: &fast_rope_impl/5, opts: new_opts}
          else
            node
          end

        other ->
          other
      end)
    end

    # Mirrors the inv_frequency branches of
    # Bumblebee.Layers.create_sinusoidal_positions/5. Returns `nil` when
    # the frequencies follow from `base` alone (the default schedule and
    # `:dynamic`, which only changes `base`), a tensor for a fixed
    # schedule, or `{short, long}` tensors for `:longrope`, which picks
    # one per call by sequence length.
    #
    # Everything is built on Nx.BinaryBackend so the tensors can be
    # embedded in the traced defn as constants (see `freqs_constant/2`).
    defp precompute_freqs(opts) do
      Nx.with_default_backend(Nx.BinaryBackend, fn -> scaled_inv_frequency(opts) end)
    end

    defp scaled_inv_frequency(opts) do
      dims = opts[:size]
      base = opts[:base] || 10_000

      case Keyword.get(opts, :scaling_strategy) do
        %{type: :linear, factor: factor} ->
          # Bumblebee divides positions by `factor`, which is the same
          # angle as dividing the inverse frequencies by it.
          default_inv_frequency(base, dims) |> Nx.divide(factor)

        %{type: :longrope, short_factor: short_factor, long_factor: long_factor} ->
          inv_frequency = default_inv_frequency(base, dims)

          {Nx.divide(
             inv_frequency,
             Nx.tensor(short_factor, type: :f32)
           ), Nx.divide(inv_frequency, Nx.tensor(long_factor, type: :f32))}

        %{
          type: :llama3,
          factor: factor,
          low_frequency_factor: low_freq_factor,
          high_frequency_factor: high_freq_factor,
          original_max_positions: orig_max_pos
        } ->
          llama3_inv_frequency(
            default_inv_frequency(base, dims),
            factor,
            low_freq_factor,
            high_freq_factor,
            orig_max_pos
          )

        _other ->
          nil
      end
    end

    defp default_inv_frequency(base, dims) do
      range = Nx.iota({div(dims, 2)}, type: :f32) |> Nx.multiply(2) |> Nx.divide(dims)
      Nx.divide(1.0, Nx.pow(base, range))
    end

    @doc false
    def fast_rope_impl(query, key, position_ids, attention_mask, opts \\ []) do
      opts =
        Keyword.validate!(opts, [
          :size,
          :scaling_strategy,
          :precomputed_freqs,
          mode: :inference,
          max_positions: 2048,
          base: 10_000
        ])

      # Same rule as Bumblebee: during cached decode position_ids only
      # covers the new tokens, but the attention mask spans the whole
      # sequence. The `:dynamic` and `:longrope` schedules depend on it.
      sequence_length =
        case attention_mask do
          %Axon.None{} -> Nx.axis_size(position_ids, 1)
          _ -> Nx.axis_size(attention_mask, 1)
        end

      {schedule, amplitude} = rope_schedule(opts, sequence_length)
      dims = opts[:size]

      rotate =
        case Nx.shape(position_ids) do
          {batch, 1} ->
            # One token per row: each row's position is exactly a
            # per-batch offset for the fused kernel.
            offset = Nx.reshape(position_ids, {batch})
            &fused_rope(&1, offset, schedule, dims)

          _ ->
            &explicit_rope(&1, position_ids, schedule, dims)
        end

      # Bumblebee's `:longrope` scales cos and sin by `amplitude`, which
      # scales the rotated output by the same factor.
      scale = fn x -> if amplitude == 1.0, do: x, else: Nx.multiply(x, amplitude) end

      {scale.(rotate.(query)), scale.(rotate.(key))}
    end

    # Resolves the frequency schedule for one call, as
    # `{{:base, base} | {:freqs, inv_freq}, amplitude}`.
    defp rope_schedule(opts, sequence_length) do
      dims = opts[:size]
      max_positions = opts[:max_positions]

      case {opts[:scaling_strategy], opts[:precomputed_freqs]} do
        {%{type: :dynamic, factor: factor}, _} when sequence_length > max_positions ->
          # Mirrors Bumblebee's base adjustment for sequences longer
          # than max_positions, including its operator order.
          base =
            :math.pow(
              opts[:base] * (factor * sequence_length / max_positions - (factor - 1)),
              dims / (dims - 2)
            )

          {{:base, base}, 1.0}

        {%{type: :longrope} = strategy, {short_freqs, long_freqs}} ->
          freqs =
            if sequence_length > strategy.original_max_positions,
              do: long_freqs,
              else: short_freqs

          {{:freqs, freqs},
           longrope_cos_sin_factor(max_positions, strategy.original_max_positions)}

        {_strategy, nil} ->
          {{:base, opts[:base]}, 1.0}

        {_strategy, freqs} ->
          {{:freqs, freqs}, 1.0}
      end
    end

    defp longrope_cos_sin_factor(max_positions, original_max_positions) do
      scale = max_positions / original_max_positions

      if scale <= 1.0 do
        1.0
      else
        :math.sqrt(:math.log(scale) / :math.log(original_max_positions) + 1.0)
      end
    end

    # Fused kernel on a {batch, seq, heads, head_dim} tensor. It takes
    # positions from axis -2, hence the transposes.
    defp fused_rope(x, offset, schedule, dims) do
      x = Nx.transpose(x, axes: [0, 2, 1, 3])

      rotated =
        case schedule do
          {:base, base} ->
            Emily.Fast.rope(x, offset, dims: dims, traditional: false, base: base * 1.0)

          {:freqs, freqs} ->
            Emily.Fast.rope_with_freqs(x, offset, freqs_constant(freqs, x),
              dims: dims,
              traditional: false
            )
        end

      Nx.transpose(rotated, axes: [0, 2, 1, 3])
    end

    # Bumblebee's rotation at arbitrary per-token positions, computing
    # the angles for the given positions only rather than a table over
    # the whole sequence.
    defp explicit_rope(x, position_ids, schedule, dims) do
      inv_freq =
        case schedule do
          {:base, base} ->
            Nx.with_default_backend(Nx.BinaryBackend, fn -> default_inv_frequency(base, dims) end)

          {:freqs, freqs} ->
            freqs
        end

      angle =
        position_ids
        |> Nx.as_type(:f32)
        |> Nx.new_axis(-1)
        |> Nx.multiply(freqs_constant(inv_freq, x))

      angle = Nx.concatenate([angle, angle], axis: -1) |> Nx.new_axis(2)
      cos = angle |> Nx.cos() |> Nx.as_type(Nx.type(x))
      sin = angle |> Nx.sin() |> Nx.as_type(Nx.type(x))

      half = div(Nx.axis_size(x, -1), 2)
      x1 = Nx.slice_along_axis(x, 0, half, axis: -1)
      x2 = Nx.slice_along_axis(x, half, half, axis: -1)
      rotated_half = Nx.concatenate([Nx.negate(x2), x1], axis: -1)

      Nx.add(Nx.multiply(x, cos), Nx.multiply(rotated_half, sin))
    end

    # The precomputed freqs are a concrete tensor, but Axon calls the
    # layer impl while tracing, where every operand must be an
    # expression.
    defp freqs_constant(freqs, %Nx.Tensor{data: %Expr{}}), do: Expr.tensor(freqs)
    defp freqs_constant(freqs, _x), do: freqs

    # Reimplementation of Bumblebee.Layers.llama3_inv_frequency/5 that
    # works on a concrete tensor (not a defn'd one). It runs Elixir-side
    # at shim-rewrite time, so concrete arithmetic is fine.
    defp llama3_inv_frequency(inv_freq, factor, low_freq_factor, high_freq_factor, orig_max_pos) do
      low_wavelength = orig_max_pos / low_freq_factor
      high_wavelength = orig_max_pos / high_freq_factor

      inv_freq_list = inv_freq |> Nx.as_type(:f32) |> Nx.to_flat_list()

      scaled =
        Enum.map(inv_freq_list, fn iv ->
          wavelength = 2 * :math.pi() / iv

          cond do
            wavelength < high_wavelength ->
              iv

            wavelength > low_wavelength ->
              iv / factor

            true ->
              smooth =
                (orig_max_pos / wavelength - low_freq_factor) /
                  (high_freq_factor - low_freq_factor)

              (1 - smooth) * iv / factor + smooth * iv
          end
        end)

      Nx.tensor(scaled, type: :f32)
    end

    # ----------------------------------------------------------------
    # SDPA
    # ----------------------------------------------------------------
    #
    # Bumblebee's attention/8 splits across two Axon.layer nodes:
    #
    #   weights = Axon.layer(&attention_weights_impl/7,
    #               [Q, K, key_mask?, head_mask?, bias?, offset?], opts)
    #   weights = Axon.dropout(weights, rate: opts[:dropout_rate])
    #   output  = Axon.layer(&attention_output_impl/3, [weights, V], opts)
    #   {output, weights}
    #
    # We rewrite the *output* node by walking back through the graph to
    # find the weights node's inputs (Q, K, mask, …) and constructing a
    # new layer that consumes [Q, K, V, masks…] directly. The original
    # weights node remains in the graph (it's referenced by the outer
    # `{output, weights}` tuple) — we accept that cost; eliminating it
    # would require model-level surgery on the output-tuple itself,
    # which `Axon.rewrite_nodes` doesn't expose. In practice Bumblebee
    # inference doesn't surface `weights` to users, so the JIT compiler
    # may dead-code-eliminate it; if not, the fused output still wins.

    @doc false
    def apply_attention(%Axon{} = model) do
      Axon.rewrite_nodes(model, &attention_rewriter/1)
    end

    # The rewriter: skip unless we're looking at attention_output_impl
    # with dropout disabled. Flattened into `cond` to stay within the
    # credo nesting limit.
    defp attention_rewriter(%Axon.Node{op: op, opts: attn_opts}) do
      cond do
        fn_mfa(op) != @attention_output_mfa -> :skip
        Keyword.get(attn_opts, :dropout_rate, 0.0) != 0 -> :skip
        true -> &build_fused_attention_from_inputs/2
      end
    end

    defp attention_rewriter(_), do: :skip

    defp build_fused_attention_from_inputs([weights_axon, value_axon], _output) do
      build_fused_attention(weights_axon, value_axon)
    end

    # Decompose a captured named function into `{module, name, arity}`,
    # or return `nil` for anonymous closures (and for non-function ops
    # like atom-keyed built-ins).
    #
    # Bumblebee captures its impls from inside its own module
    # (`&attention_output_impl/3`), which yields a `:local` fun, not an
    # `:external` one. `Function.info/2` still reports the real module,
    # name and arity for those; anonymous funs get a compiler-generated
    # name like `-attention/8-fun-0-`.
    defp fn_mfa(fun) when is_function(fun) do
      {:module, m} = Function.info(fun, :module)
      {:name, n} = Function.info(fun, :name)
      {:arity, a} = Function.info(fun, :arity)

      case Function.info(fun, :type) do
        {:type, :external} -> {m, n, a}
        {:type, :local} -> if anonymous_fun_name?(n), do: nil, else: {m, n, a}
      end
    end

    defp fn_mfa(_), do: nil

    defp anonymous_fun_name?(name), do: String.starts_with?(Atom.to_string(name), "-")

    # Walk back from the `weights` input (a dropout layer wrapping the
    # attention_weights_impl call) and extract the original Q/K/mask/etc
    # Axon graphs.
    defp build_fused_attention(%Axon{output: id, nodes: nodes}, value_axon) do
      weights_node = nodes[id]
      weights_id = unwrap_dropout(weights_node, nodes, id)
      %Axon.Node{parent: parent_ids, opts: weights_opts} = nodes[weights_id]

      # Bumblebee's attention_weights_impl takes a fixed 6-input list:
      # [query, key, key_mask, head_mask, bias, offset]. A version skew
      # that changes the arity would otherwise crash with an opaque
      # MatchError mid-rewrite; surface it as a named error instead.
      [q_id, k_id, km_id, hm_id, bias_id, off_id] =
        case parent_ids do
          [_, _, _, _, _, _] = ids ->
            ids

          other ->
            raise "Emily.Bumblebee.FastKernels: attention_weights_impl expected " <>
                    "6 parents, got #{length(other)}. Bumblebee version skew?"
        end

      q = %Axon{output: q_id, nodes: nodes}
      k = %Axon{output: k_id, nodes: nodes}
      key_mask = %Axon{output: km_id, nodes: nodes}
      head_mask = %Axon{output: hm_id, nodes: nodes}
      bias = %Axon{output: bias_id, nodes: nodes}
      offset = %Axon{output: off_id, nodes: nodes}

      Axon.layer(
        &fast_sdpa_impl/8,
        [q, k, value_axon, key_mask, head_mask, bias, offset],
        causal: Keyword.get(weights_opts, :causal, false),
        window_size: Keyword.get(weights_opts, :window_size),
        scale: Keyword.get(weights_opts, :scale)
      )
    end

    defp unwrap_dropout(%Axon.Node{op_name: :dropout, parent: [parent_id]}, _nodes, _id),
      do: parent_id

    defp unwrap_dropout(_node, _nodes, id), do: id

    # The fused replacement. Mirrors the input contract of
    # attention_weights_impl + attention_output_impl combined: takes Q,
    # K, V plus the same mask/bias/offset signals plus a leading no-op
    # input slot for the eighth arg (Axon.layer requires a fixed input
    # list shape — we use a dummy %Axon.None{} when there's nothing
    # interesting to thread through).
    @doc false
    def fast_sdpa_impl(query, key, value, key_mask, head_mask, bias, offset, opts \\ []) do
      opts = Keyword.validate!(opts, [:causal, :window_size, :scale, mode: :inference])

      # Layout: {batch, seq, heads, head_dim} → {batch, heads, seq, head_dim}
      q = Nx.transpose(query, axes: [0, 2, 1, 3])
      k = Nx.transpose(key, axes: [0, 2, 1, 3])
      v = Nx.transpose(value, axes: [0, 2, 1, 3])

      scale =
        case opts[:scale] do
          nil -> 1.0 / :math.sqrt(Nx.axis_size(q, -1))
          s -> s
        end

      q_seq = Nx.axis_size(q, -2)
      k_seq = Nx.axis_size(k, -2)
      type = Nx.type(q)

      mask = build_attention_mask(key_mask, bias, offset, opts, q_seq, k_seq, type)
      sdpa_opts = [scale: scale]

      out_bhsd =
        case mask do
          :none ->
            Emily.Fast.scaled_dot_product_attention(q, k, v, [causal: false] ++ sdpa_opts)

          :causal ->
            Emily.Fast.scaled_dot_product_attention(q, k, v, [causal: true] ++ sdpa_opts)

          %Nx.Tensor{} = additive ->
            Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, additive, sdpa_opts)
        end

      out = Nx.transpose(out_bhsd, axes: [0, 2, 1, 3])

      case head_mask do
        %Axon.None{} ->
          out

        _ ->
          head_mask = Nx.reshape(head_mask, {1, :auto, 1, 1})
          # Bumblebee scales the per-head weights by head_mask before
          # multiplying by V. Scaling the per-head outputs instead is
          # equivalent, since the weights-by-V product is linear.
          out
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.multiply(head_mask)
          |> Nx.transpose(axes: [0, 2, 1, 3])
      end
    end

    # offset comes through as %Axon.None{} or as a tensor.
    defp ensure_offset_scalar(%Axon.None{}), do: 0
    defp ensure_offset_scalar(t), do: t

    # Replicates the mask-construction in attention_weights_impl.
    # Returns one of :none, :causal, or an additive %Nx.Tensor{}.
    defp build_attention_mask(key_mask, bias, offset, opts, q_seq, k_seq, type) do
      case classify_mask(key_mask, bias, offset, opts, q_seq, k_seq) do
        :none ->
          :none

        :causal ->
          :causal

        :array ->
          offset = ensure_offset_scalar(offset)
          additive_mask(key_mask, bias, offset, opts, q_seq, k_seq, type)
      end
    end

    # MLX's built-in causal mask aligns the last query with the last
    # key and knows nothing about Bumblebee's cache `offset`. It only
    # matches Bumblebee's causal mask (query i sees keys 0..offset+i)
    # when there is no offset and Q and K have the same length; in every
    # other causal case, such as decoding against a preallocated KV
    # cache, we build the mask explicitly.
    defp classify_mask(key_mask, bias, offset, opts, q_seq, k_seq) do
      plain? = none?(key_mask) and none?(bias) and opts[:window_size] == nil
      aligned? = none?(offset) and q_seq == k_seq

      cond do
        plain? and opts[:causal] != true -> :none
        plain? and aligned? -> :causal
        true -> :array
      end
    end

    # Absent optional inputs arrive as `%Axon.None{__propagate__: false}`,
    # so match on the struct rather than comparing to `%Axon.None{}`.
    defp none?(%Axon.None{}), do: true
    defp none?(_), do: false

    defp additive_mask(key_mask, bias, offset, opts, q_seq, k_seq, type) do
      key_mask_tensor = resolve_key_mask(key_mask)
      causal_window_mask = resolve_causal_window(opts, q_seq, k_seq, offset)
      keep_mask = Nx.logical_and(key_mask_tensor, causal_window_mask)
      apply_bias_mask(keep_mask, bias, type)
    end

    defp resolve_key_mask(%Axon.None{}),
      do: Nx.broadcast(Nx.tensor(1, type: {:u, 8}), {1, 1, 1, 1})

    defp resolve_key_mask(t), do: coerce_key_mask(t)

    defp resolve_causal_window(opts, q_seq, k_seq, offset) do
      case {opts[:causal], opts[:window_size]} do
        {false, nil} ->
          Nx.broadcast(Nx.tensor(1, type: {:u, 8}), {1, 1})

        {true, nil} ->
          causal_mask_tensor(q_seq, k_seq, offset)

        {false, {left, right}} ->
          window_mask_tensor(q_seq, k_seq, offset, left, right)

        {true, {left, _right}} ->
          window_mask_tensor(q_seq, k_seq, offset, left, 0)
      end
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
    end

    defp apply_bias_mask(keep_mask, %Axon.None{}, type) do
      Nx.select(keep_mask, Nx.tensor(0.0, type: type), Nx.Constants.min_finite(type))
    end

    defp apply_bias_mask(keep_mask, bias_tensor, type) do
      Nx.select(
        Nx.broadcast(keep_mask, broadcast_target(keep_mask, bias_tensor)),
        bias_tensor,
        Nx.Constants.min_finite(type)
      )
    end

    defp coerce_key_mask(t) do
      case Nx.rank(t) do
        2 -> t |> Nx.new_axis(1) |> Nx.new_axis(1)
        4 -> t
      end
    end

    defp causal_mask_tensor(q_len, k_len, offset) do
      Nx.greater_equal(
        Nx.add(Nx.iota({q_len, 1}), offset),
        Nx.iota({1, k_len})
      )
    end

    defp window_mask_tensor(q_len, k_len, offset, left, right) do
      diff =
        Nx.subtract(
          Nx.add(Nx.iota({q_len, 1}), offset),
          Nx.iota({1, k_len})
        )

      Nx.logical_and(Nx.less_equal(diff, left), Nx.greater_equal(diff, -right))
    end

    defp broadcast_target(a, b) do
      a_shape = Nx.shape(a) |> Tuple.to_list()
      b_shape = Nx.shape(b) |> Tuple.to_list()

      [a_shape, b_shape]
      |> Enum.map(&Enum.reverse/1)
      |> Enum.zip()
      |> Enum.map(fn {x, y} -> max(x, y) end)
      |> Enum.reverse()
      |> List.to_tuple()
    end
  end
end
