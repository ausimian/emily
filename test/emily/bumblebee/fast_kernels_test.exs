defmodule Emily.Bumblebee.FastKernelsTest do
  @moduledoc """
  Unit tests for `Emily.Bumblebee.FastKernels`. Each test builds a
  handcrafted Axon model containing one of the patterns the shim
  knows how to rewrite, runs `apply/1`, then verifies:

    1. The rewritten model's predict_fn produces output within
       tolerance of the unrewritten model.
    2. The expected `op` field on the relevant node has been swapped
       to the fast variant.

  Conformance against full Bumblebee transformer models lives in
  `test/emily/conformance/*_full_test.exs` (opt-in via the
  `:fast_kernels_full` tag).
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3]

  alias Bumblebee.Layers, as: BL
  alias Emily.Bumblebee.FastKernels

  @f32_tol 1.0e-4

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

  describe "apply_rms_norm/1" do
    test "rewrites BL.rms_norm and matches output" do
      model =
        Axon.input("x", shape: {nil, 4, 16})
        |> BL.rms_norm(name: "norm", epsilon: 1.0e-6)

      rewritten = FastKernels.apply_rms_norm(model)

      # Same {init, predict}.
      {init_fn, predict_fn_orig} = Axon.build(model, compiler: Emily.Compiler)
      {_, predict_fn_fast} = Axon.build(rewritten, compiler: Emily.Compiler)

      x = Nx.iota({1, 4, 16}, type: :f32, backend: Emily.Backend) |> Nx.divide(50)
      params = init_fn.(%{"x" => x}, Axon.ModelState.empty())

      orig = predict_fn_orig.(params, %{"x" => x})
      fast = predict_fn_fast.(params, %{"x" => x})

      assert_close(fast, orig, tol: @f32_tol)
    end
  end

  describe "apply_layer_norm/1" do
    test "rewrites Axon.layer_norm and matches output" do
      model =
        Axon.input("x", shape: {nil, 4, 16})
        |> Axon.layer_norm(name: "norm", epsilon: 1.0e-5)

      rewritten = FastKernels.apply_layer_norm(model)

      {init_fn, predict_orig} = Axon.build(model, compiler: Emily.Compiler)
      {_, predict_fast} = Axon.build(rewritten, compiler: Emily.Compiler)

      x = Nx.iota({1, 4, 16}, type: :f32, backend: Emily.Backend) |> Nx.divide(50)
      params = init_fn.(%{"x" => x}, Axon.ModelState.empty())

      orig = predict_orig.(params, %{"x" => x})
      fast = predict_fast.(params, %{"x" => x})

      assert_close(fast, orig, tol: @f32_tol)
    end
  end

  # The fallback lane runs the rewritten model on Nx.BinaryBackend, where
  # every `Emily.Fast` call takes its composed-defn fallback.
  @lanes [
    default: {Emily.Backend, [compiler: Emily.Compiler]},
    native: {Emily.Backend, [compiler: Emily.Compiler, native: true, native_fallback: :raise]},
    fallback: {Nx.BinaryBackend, [compiler: Nx.Defn.Evaluator]}
  ]

  describe "apply_rope/1" do
    @rope_strategies [
      default: nil,
      linear: %{type: :linear, factor: 2.0},
      dynamic: %{type: :dynamic, factor: 2.0},
      longrope: %{
        type: :longrope,
        short_factor: [1.0, 1.5, 2.0, 3.0],
        long_factor: [2.0, 3.0, 4.0, 6.0],
        original_max_positions: 6
      },
      llama3: %{
        type: :llama3,
        factor: 8.0,
        low_frequency_factor: 1.0,
        high_frequency_factor: 4.0,
        original_max_positions: 16
      }
    ]

    # `max_positions: 8` with sequence lengths of 4 and 12 exercises
    # both sides of the `:dynamic` and `:longrope` length thresholds.
    # Rows get different, partly non-contiguous positions, as Bumblebee
    # generation produces for padded batches: prefill takes the
    # explicit-position path, single-token decode the fused kernel with
    # per-row offsets.
    for {name, strategy} <- @rope_strategies,
        len <- [4, 12],
        {lane, {backend, compiler}} <- @lanes do
      test "rewrites rotary_embedding (#{name}, prefill #{len}, #{lane}) and matches output" do
        Nx.default_backend(unquote(backend))
        model = rope_model(unquote(Macro.escape(strategy)))
        rewritten = FastKernels.apply_rope(model)

        assert op_names(rewritten) |> Enum.member?(:fast_rope_impl)
        refute op_names(rewritten) |> Enum.member?(:apply_rotary_embedding)

        # The positions Bumblebee generation derives from the attention
        # mask: row 0 is left-padded by two tokens, row 1 right-padded.
        iota = Nx.iota({unquote(len)}, type: :s64)

        inputs = %{
          "q" => random({2, unquote(len), 2, 8}, 0),
          "k" => random({2, unquote(len), 2, 8}, 1),
          "position_ids" =>
            Nx.stack([Nx.max(Nx.subtract(iota, 2), 0), Nx.min(iota, unquote(len) - 3)])
        }

        assert_same_tuple_output(model, rewritten, inputs, unquote(compiler))
      end

      test "rewrites rotary_embedding (#{name}, decode at #{len}, #{lane}) and matches output" do
        Nx.default_backend(unquote(backend))
        model = rope_model(unquote(Macro.escape(strategy)), attention_mask: true)
        rewritten = FastKernels.apply_rope(model)

        # One new token per row, at different positions. The mask spans
        # the whole sequence, which is what the length-dependent
        # schedules key on.
        inputs = %{
          "q" => random({2, 1, 2, 8}, 0),
          "k" => random({2, 1, 2, 8}, 1),
          "position_ids" => Nx.tensor([[unquote(len) - 3], [unquote(len) - 1]], type: :s64),
          "attention_mask" => Nx.broadcast(Nx.tensor(1, type: :s64), {2, unquote(len)})
        }

        assert_same_tuple_output(model, rewritten, inputs, unquote(compiler))
      end
    end
  end

  describe "apply_attention/2" do
    @attention_cases [
      plain: [],
      causal: [causal: true],
      window: [window_size: {2, 1}],
      causal_window: [causal: true, window_size: {2, 2}],
      scale: [scale: 0.5]
    ]

    for {name, opts} <- @attention_cases, {lane, {backend, compiler}} <- @lanes do
      test "rewrites attention (#{name}, #{lane}) and matches output" do
        Nx.default_backend(unquote(backend))
        model = attention_model(unquote(opts))
        rewritten = FastKernels.apply_attention(model)

        assert op_names(rewritten) |> Enum.member?(:fast_sdpa_impl)
        refute op_names(rewritten) |> Enum.member?(:attention_output_impl)

        inputs = %{
          "q" => random({2, 6, 2, 8}, 0),
          "k" => random({2, 6, 2, 8}, 1),
          "v" => random({2, 6, 2, 8}, 2)
        }

        assert_same_output(model, rewritten, inputs, unquote(compiler))
      end
    end

    for {lane, {backend, compiler}} <- @lanes do
      test "matches causal decode against a longer KV cache with key mask (#{lane})" do
        Nx.default_backend(unquote(backend))
        model = attention_model([causal: true], key_mask: true, offset: true)
        rewritten = FastKernels.apply_attention(model)

        # One query at position 3 attending over an 8-slot cache with
        # slots beyond the current position masked off.
        inputs = %{
          "q" => random({1, 1, 2, 8}, 0),
          "k" => random({1, 8, 2, 8}, 1),
          "v" => random({1, 8, 2, 8}, 2),
          "key_mask" => Nx.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], type: :s64),
          "offset" => Nx.tensor(3, type: :s64)
        }

        assert_same_output(model, rewritten, inputs, unquote(compiler))
      end

      test "matches causal attention with an offset and no key mask (#{lane})" do
        Nx.default_backend(unquote(backend))
        model = attention_model([causal: true], offset: true)
        rewritten = FastKernels.apply_attention(model)

        # Without a key mask the cache offset alone must hide slots past
        # the current position; MLX's built-in causal mask would not.
        inputs = %{
          "q" => random({1, 2, 2, 8}, 0),
          "k" => random({1, 8, 2, 8}, 1),
          "v" => random({1, 8, 2, 8}, 2),
          "offset" => Nx.tensor(3, type: :s64)
        }

        assert_same_output(model, rewritten, inputs, unquote(compiler))
      end

      test "matches attention with bias and head mask (#{lane})" do
        Nx.default_backend(unquote(backend))
        model = attention_model([], bias: true, head_mask: true)
        rewritten = FastKernels.apply_attention(model)

        inputs = %{
          "q" => random({1, 4, 2, 8}, 0),
          "k" => random({1, 4, 2, 8}, 1),
          "v" => random({1, 4, 2, 8}, 2),
          "bias" => random({1, 2, 4, 4}, 3),
          "head_mask" => Nx.tensor([1.0, 0.25])
        }

        assert_same_output(model, rewritten, inputs, unquote(compiler))
      end
    end

    test "leaves attention with dropout in place" do
      model = attention_model(dropout_rate: 0.1)
      rewritten = FastKernels.apply_attention(model)

      assert op_names(rewritten) |> Enum.member?(:attention_output_impl)
      refute op_names(rewritten) |> Enum.member?(:fast_sdpa_impl)
    end
  end

  describe "apply/2 :force_fused_sdpa" do
    test "defaults to false on the rewritten attention layers" do
      assert fused_sdpa_opts(FastKernels.apply(attention_model([]))) == [false]
    end

    test "passes force_fused: true to the rewritten attention layers" do
      # MLX's fused kernels need a head dim of 64 or more.
      model = attention_model([], head_dim: 64)
      rewritten = FastKernels.apply(model, force_fused_sdpa: true)

      assert fused_sdpa_opts(rewritten) == [true]

      inputs = %{
        "q" => random({1, 4, 2, 64}, 0),
        "k" => random({1, 4, 2, 64}, 1),
        "v" => random({1, 4, 2, 64}, 2)
      }

      assert_same_output(model, rewritten, inputs, compiler: Emily.Compiler)
    end

    test "rejects unknown options" do
      assert_raise ArgumentError, fn -> FastKernels.apply(attention_model([]), force: true) end
    end
  end

  describe "apply/1 idempotence" do
    test "applying twice doesn't break the model" do
      model =
        Axon.input("x", shape: {nil, 4, 16})
        |> BL.rms_norm(name: "n1", epsilon: 1.0e-6)
        |> Axon.layer_norm(name: "n2", epsilon: 1.0e-5)

      once = FastKernels.apply(model)
      twice = FastKernels.apply(once)

      {init_fn, predict_once} = Axon.build(once, compiler: Emily.Compiler)
      {_, predict_twice} = Axon.build(twice, compiler: Emily.Compiler)

      x = Nx.iota({1, 4, 16}, type: :f32, backend: Emily.Backend) |> Nx.divide(50)
      params = init_fn.(%{"x" => x}, Axon.ModelState.empty())

      a = predict_once.(params, %{"x" => x})
      b = predict_twice.(params, %{"x" => x})

      assert_close(a, b, tol: @f32_tol)
    end
  end

  defp rope_model(strategy, opts \\ []) do
    q = Axon.input("q", shape: {nil, nil, 2, 8})
    k = Axon.input("k", shape: {nil, nil, 2, 8})
    position_ids = Axon.input("position_ids", shape: {nil, nil})

    attention_mask =
      if opts[:attention_mask],
        do: Axon.input("attention_mask", shape: {nil, nil}),
        else: BL.none()

    rope_opts =
      [max_positions: 8, base: 10_000] ++ if(strategy, do: [scaling_strategy: strategy], else: [])

    {q, k} = BL.rotary_embedding(q, k, position_ids, attention_mask, 8, rope_opts)
    Axon.container({q, k})
  end

  defp attention_model(attention_opts, inputs \\ []) do
    optional = fn name, shape ->
      if inputs[name], do: Axon.input(Atom.to_string(name), shape: shape), else: BL.none()
    end

    head_dim = Keyword.get(inputs, :head_dim, 8)
    q = Axon.input("q", shape: {nil, nil, 2, head_dim})
    k = Axon.input("k", shape: {nil, nil, 2, head_dim})
    v = Axon.input("v", shape: {nil, nil, 2, head_dim})

    {out, _weights} =
      BL.attention(
        q,
        k,
        v,
        optional.(:key_mask, {nil, nil}),
        optional.(:head_mask, {2}),
        optional.(:bias, {nil, 2, nil, nil}),
        optional.(:offset, {}),
        attention_opts
      )

    out
  end

  defp op_names(%Axon{nodes: nodes}) do
    for {_id, %Axon.Node{op: op}} <- nodes, is_function(op) do
      {:name, name} = Function.info(op, :name)
      name
    end
  end

  defp fused_sdpa_opts(%Axon{nodes: nodes}) do
    for {_id, %Axon.Node{op: op, opts: opts}} <- nodes,
        is_function(op),
        Function.info(op, :name) == {:name, :fast_sdpa_impl},
        do: opts[:force_fused]
  end

  defp random(shape, seed) do
    key = Nx.Random.key(seed)
    {t, _key} = Nx.Random.normal(key, shape: shape, type: :f32)
    t
  end

  defp predict_both(model, rewritten, inputs, compiler_opts) do
    {init_fn, predict_orig} = Axon.build(model, compiler_opts)
    {_, predict_fast} = Axon.build(rewritten, compiler_opts)

    params = init_fn.(inputs, Axon.ModelState.empty())
    {predict_orig.(params, inputs), predict_fast.(params, inputs)}
  end

  defp assert_same_output(model, rewritten, inputs, compiler_opts) do
    {orig, fast} = predict_both(model, rewritten, inputs, compiler_opts)
    assert_close(fast, orig, tol: @f32_tol)
  end

  defp assert_same_tuple_output(model, rewritten, inputs, compiler_opts) do
    {{orig_q, orig_k}, {fast_q, fast_k}} = predict_both(model, rewritten, inputs, compiler_opts)
    assert_close(fast_q, orig_q, tol: @f32_tol)
    assert_close(fast_k, orig_k, tol: @f32_tol)
  end
end
