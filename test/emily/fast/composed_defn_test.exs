defmodule Emily.Fast.ComposedDefnTest do
  @moduledoc """
  Regression tests for #205: every defn-callable `Emily.Fast` kernel
  must be callable from *inside a caller's own defn* — genuine
  composition — not just as the literal top-level unit handed to
  `Nx.Defn.jit_apply`. Each caller below nests the kernel two levels
  deep (`defn` → `defnp` → kernel), the exact shape that used to raise
  "cannot invoke ... inside defn because it was not defined with defn".

  Three axes of coverage:

    * **Repro** — the issue's own two-level nesting runs and produces
      the right shape/values, including under global
      `Emily.Backend`/`Emily.Compiler` defaults.
    * **Conformance** — for every converted kernel, the composed result
      under `Emily.Compiler` (fused `mx::fast::*` path) matches the
      composed result under `Nx.Defn.Evaluator` + `Nx.BinaryBackend`
      (composed-defn fallback) within f32 tolerance.
    * **Fused path fires** — positively: the composed graph lowers to
      the `fast_*` IR opcodes (native lane), and the conformance runs
      pass `native_fallback: :raise` so an Evaluator fallback fails
      loudly; negatively: `[:emily, :block, :fallback]` never fires for
      an `Emily.Fast.Block.*` struct on the `native: false` Evaluator
      walk (which dispatches through `Emily.Backend.block/4`).
  """

  # One test mutates the global default backend/compiler; the telemetry
  # tests attach process-global handlers.
  use ExUnit.Case, async: false

  import Nx.Defn
  import Emily.BackendGenerators, only: [assert_close: 3]

  alias Emily.IR
  alias Nx.Defn.Composite

  @f32_tol 1.0e-5

  # ----------------------------------------------------------------
  # Composed callers: defn -> defnp -> fused kernel (two levels deep)
  # ----------------------------------------------------------------

  defn rms_caller(x, w) do
    rms_inner(x, w)
  end

  defnp rms_inner(x, w) do
    Emily.Fast.rms_norm(x, w, eps: 1.0e-5)
  end

  defn layer_norm_caller(x, w, b) do
    layer_norm_inner(x, w, b)
  end

  defnp layer_norm_inner(x, w, b) do
    Emily.Fast.layer_norm(x, w, b, eps: 1.0e-5)
  end

  defn rope_caller(x, o) do
    rope_inner(x, o)
  end

  defnp rope_inner(x, o) do
    Emily.Fast.rope(x, o, dims: 8)
  end

  defn rope_freqs_caller(x, o, f) do
    rope_freqs_inner(x, o, f)
  end

  defnp rope_freqs_inner(x, o, f) do
    Emily.Fast.rope_with_freqs(x, o, f, dims: 8)
  end

  defn sdpa_caller(q, k, v) do
    sdpa_inner(q, k, v)
  end

  defnp sdpa_inner(q, k, v) do
    Emily.Fast.scaled_dot_product_attention(q, k, v, causal: true)
  end

  defn sdpa_sinks_caller(q, k, v, s) do
    sdpa_sinks_inner(q, k, v, s)
  end

  defnp sdpa_sinks_inner(q, k, v, s) do
    Emily.Fast.scaled_dot_product_attention(q, k, v, sinks: s)
  end

  defn sdpa_mask_caller(q, k, v, m) do
    sdpa_mask_inner(q, k, v, m)
  end

  defnp sdpa_mask_inner(q, k, v, m) do
    Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, m)
  end

  defn sdpa_mask_sinks_caller(q, k, v, m, s) do
    sdpa_mask_sinks_inner(q, k, v, m, s)
  end

  defnp sdpa_mask_sinks_inner(q, k, v, m, s) do
    Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, m, sinks: s)
  end

  # A caller mixing a fused kernel with regular Nx ops, still two deep.
  defn mixed_caller(x, w) do
    mixed_inner(x, w)
  end

  defnp mixed_inner(x, w) do
    x
    |> Nx.multiply(2.0)
    |> Emily.Fast.rms_norm(w, eps: 1.0e-5)
    |> Nx.add(1.0)
  end

  # ----------------------------------------------------------------
  # Shared inputs (BinaryBackend refs; copied to Emily per test)
  # ----------------------------------------------------------------

  defp norm_inputs do
    x = Nx.iota({2, 8, 32}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(100)
    w = Nx.iota({32}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(1) |> Nx.divide(32)
    b = Nx.iota({32}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(64)
    {x, w, b}
  end

  defp rope_inputs do
    x = Nx.iota({1, 2, 4, 8}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(10)
    o = Nx.tensor(3, backend: Nx.BinaryBackend)
    f = Nx.tensor([1.0, 0.1, 0.01, 0.001], backend: Nx.BinaryBackend)
    {x, o, f}
  end

  defp sdpa_inputs do
    q = Nx.iota({1, 2, 4, 8}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(50)
    k = Nx.iota({1, 2, 4, 8}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(40)
    v = Nx.iota({1, 2, 4, 8}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(30)
    m = Nx.iota({1, 1, 4, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(-8)
    s = Nx.tensor([0.5, -0.25], backend: Nx.BinaryBackend)
    {q, k, v, m, s}
  end

  defp to_emily(tensors), do: Enum.map(tensors, &Nx.backend_copy(&1, Emily.Backend))

  # Composed result on the fused lane. `native_fallback: :raise` makes a
  # silent Evaluator fallback fail loudly — the composed graph must
  # lower fully native.
  defp fused(caller, args) do
    Nx.Defn.jit_apply(caller, to_emily(args),
      compiler: Emily.Compiler,
      native: true,
      native_fallback: :raise
    )
  end

  # Composed result on the fallback lane: plain Evaluator + BinaryBackend.
  defp fallback(caller, args) do
    Nx.Defn.jit_apply(caller, args, compiler: Nx.Defn.Evaluator)
  end

  # ----------------------------------------------------------------
  # §1 — the issue's repro, two levels deep
  # ----------------------------------------------------------------

  describe "issue #205 repro" do
    test "rms_norm composes inside a caller's own defn under global defaults" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], backend: Emily.Backend)
      w = Nx.tensor([1.0, 1.0, 1.0, 1.0], backend: Emily.Backend)

      prev_backend = Nx.global_default_backend(Emily.Backend)
      prev_options = Nx.Defn.global_default_options(compiler: Emily.Compiler)

      on_exit(fn ->
        Nx.global_default_backend(prev_backend)
        Nx.Defn.global_default_options(prev_options)
      end)

      result = rms_caller(x, w)
      assert Nx.shape(result) == {1, 4}

      expected =
        fallback(&rms_caller/2, [
          Nx.backend_copy(x, Nx.BinaryBackend),
          Nx.backend_copy(w, Nx.BinaryBackend)
        ])

      assert_close(result, expected, tol: @f32_tol)
    end

    test "rope composes inside a caller's own defn" do
      {x, o, _f} = rope_inputs()

      result = fused(&rope_caller/2, [x, o])
      assert Nx.shape(result) == {1, 2, 4, 8}
      assert_close(result, fallback(&rope_caller/2, [x, o]), tol: @f32_tol)
    end

    test "scaled_dot_product_attention_with_mask composes inside a caller's own defn" do
      {q, k, v, m, _s} = sdpa_inputs()

      result = fused(&sdpa_mask_caller/4, [q, k, v, m])
      assert Nx.shape(result) == {1, 2, 4, 8}
      assert_close(result, fallback(&sdpa_mask_caller/4, [q, k, v, m]), tol: @f32_tol)
    end
  end

  # ----------------------------------------------------------------
  # §2 — conformance: fused lane vs fallback lane, per kernel
  # ----------------------------------------------------------------

  describe "composed conformance (Emily.Compiler vs Evaluator+BinaryBackend)" do
    test "rms_norm" do
      {x, w, _b} = norm_inputs()
      assert_close(fused(&rms_caller/2, [x, w]), fallback(&rms_caller/2, [x, w]), tol: @f32_tol)
    end

    test "layer_norm" do
      {x, w, b} = norm_inputs()

      assert_close(
        fused(&layer_norm_caller/3, [x, w, b]),
        fallback(&layer_norm_caller/3, [x, w, b]),
        tol: @f32_tol
      )
    end

    test "rope" do
      {x, o, _f} = rope_inputs()
      assert_close(fused(&rope_caller/2, [x, o]), fallback(&rope_caller/2, [x, o]), tol: @f32_tol)
    end

    test "rope_with_freqs" do
      {x, o, f} = rope_inputs()

      assert_close(
        fused(&rope_freqs_caller/3, [x, o, f]),
        fallback(&rope_freqs_caller/3, [x, o, f]),
        tol: @f32_tol
      )
    end

    test "scaled_dot_product_attention (causal)" do
      {q, k, v, _m, _s} = sdpa_inputs()

      assert_close(
        fused(&sdpa_caller/3, [q, k, v]),
        fallback(&sdpa_caller/3, [q, k, v]),
        tol: @f32_tol
      )
    end

    test "scaled_dot_product_attention with sinks" do
      {q, k, v, _m, s} = sdpa_inputs()

      assert_close(
        fused(&sdpa_sinks_caller/4, [q, k, v, s]),
        fallback(&sdpa_sinks_caller/4, [q, k, v, s]),
        tol: @f32_tol
      )
    end

    test "scaled_dot_product_attention_with_mask" do
      {q, k, v, m, _s} = sdpa_inputs()

      assert_close(
        fused(&sdpa_mask_caller/4, [q, k, v, m]),
        fallback(&sdpa_mask_caller/4, [q, k, v, m]),
        tol: @f32_tol
      )
    end

    test "scaled_dot_product_attention_with_mask with sinks" do
      {q, k, v, m, s} = sdpa_inputs()

      assert_close(
        fused(&sdpa_mask_sinks_caller/5, [q, k, v, m, s]),
        fallback(&sdpa_mask_sinks_caller/5, [q, k, v, m, s]),
        tol: @f32_tol
      )
    end

    test "fused kernel mixed with regular Nx ops" do
      {x, w, _b} = norm_inputs()

      assert_close(
        fused(&mixed_caller/2, [x, w]),
        fallback(&mixed_caller/2, [x, w]),
        tol: @f32_tol
      )
    end
  end

  # ----------------------------------------------------------------
  # §2 — the fused path actually fires when composed
  # ----------------------------------------------------------------

  # Trace a composed caller and lower it to the native IR, mirroring
  # Emily.Compiler's native lane.
  defp lowered_opcodes(caller, args) do
    templates = Enum.map(args, &Nx.template(Nx.shape(&1), Nx.type(&1)))
    expr = Nx.Defn.debug_expr_apply(caller, templates)

    {_template, leaves_rev} =
      Composite.traverse(expr, [], fn leaf, acc -> {leaf, [leaf | acc]} end)

    %IR{instrs: instrs} = leaves_rev |> Enum.reverse() |> IR.lower()
    Enum.map(instrs, & &1.opcode)
  end

  describe "composed graphs lower to the fused IR opcodes (native lane)" do
    test "each composed caller contains its fast_* opcode" do
      {x, w, b} = norm_inputs()
      {rx, o, f} = rope_inputs()
      {q, k, v, m, s} = sdpa_inputs()

      for {caller, args, opcode} <- [
            {&rms_caller/2, [x, w], :fast_rms_norm},
            {&layer_norm_caller/3, [x, w, b], :fast_layer_norm},
            {&rope_caller/2, [rx, o], :fast_rope},
            {&rope_freqs_caller/3, [rx, o, f], :fast_rope_freqs},
            {&sdpa_caller/3, [q, k, v], :fast_sdpa},
            {&sdpa_mask_caller/4, [q, k, v, m], :fast_sdpa_mask}
          ] do
        opcodes = lowered_opcodes(caller, args)

        assert opcode in opcodes,
               "expected composed graph to lower to #{inspect(opcode)}, got #{inspect(opcodes)}"
      end
    end
  end

  describe "no block-dispatch fallback for Emily.Fast blocks (evaluator lane)" do
    setup do
      handler_id = {__MODULE__, self()}
      parent = self()

      :telemetry.attach(
        handler_id,
        [:emily, :block, :fallback],
        fn _event, _measurements, metadata, _config ->
          send(parent, {:block_fallback, metadata.struct})
        end,
        nil
      )

      on_exit(fn -> :telemetry.detach(handler_id) end)
      :ok
    end

    test "composed kernels dispatch to Emily.Backend's native block clauses" do
      {x, w, _b} = norm_inputs()
      {q, k, v, m, _s} = sdpa_inputs()

      # native: false forces the op-by-op Evaluator walk, where every
      # Nx.block dispatches through Emily.Backend.block/4 — the FB.*
      # clauses must match, never the telemetry-emitting catch-all.
      for {caller, args} <- [
            {&rms_caller/2, [x, w]},
            {&sdpa_mask_caller/4, [q, k, v, m]}
          ] do
        Nx.Defn.jit_apply(caller, to_emily(args), compiler: Emily.Compiler, native: false)
      end

      receive do
        {:block_fallback, struct} ->
          if match?("Elixir.Emily.Fast.Block." <> _, Atom.to_string(struct)) do
            flunk("Emily.Fast block #{inspect(struct)} fell back to the composed slow path")
          end
      after
        0 -> :ok
      end
    end
  end
end
