defmodule Emily.Fast.SDPATest do
  @moduledoc """
  Tests for `Emily.Fast.scaled_dot_product_attention/4` and
  `scaled_dot_product_attention_with_mask/5`.

  Input layout: `{batch, heads, seq, head_dim}` — the canonical MLX
  and Bumblebee in-flight shape.
  """

  use ExUnit.Case, async: true

  doctest Emily.Fast,
    only: [scaled_dot_product_attention: 4, scaled_dot_product_attention_with_mask: 5]

  import Emily.BackendGenerators, only: [assert_close: 3]

  @f32_tol 1.0e-4
  @bf16_tol 1.0e-2

  setup do
    Nx.default_backend(Emily.Backend)
    :ok
  end

  defp qkv_tensors(shape, backend, type \\ :f32) do
    # Deterministic-but-not-constant inputs. Iota scales blow up
    # softmax for larger sequences, so normalise.
    size = Tuple.to_list(shape) |> Enum.reduce(&*/2)

    q =
      Nx.iota(shape, type: :f32, backend: backend)
      |> Nx.divide(size)
      |> Nx.as_type(type)

    k =
      Nx.iota(shape, type: :f32, backend: backend)
      |> Nx.add(size / 2)
      |> Nx.divide(size)
      |> Nx.as_type(type)

    v =
      Nx.iota(shape, type: :f32, backend: backend)
      |> Nx.multiply(-1.0)
      |> Nx.divide(size)
      |> Nx.as_type(type)

    {q, k, v}
  end

  describe "scaled_dot_product_attention/4 (no mask)" do
    test "fused matches defn fallback for f32" do
      shape = {1, 2, 4, 8}

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend)

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)

      fun = fn q, k, v ->
        Emily.Fast.scaled_dot_product_attention(q, k, v, causal: false)
      end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_q, ref_k, ref_v)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v)

      assert_close(fused, expected, tol: @f32_tol)
    end
  end

  describe "scaled_dot_product_attention/4 (causal mask)" do
    test "fused matches defn fallback for f32" do
      shape = {1, 2, 6, 8}

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend)

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)

      fun = fn q, k, v ->
        Emily.Fast.scaled_dot_product_attention(q, k, v, causal: true)
      end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_q, ref_k, ref_v)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v)

      assert_close(fused, expected, tol: @f32_tol)
    end
  end

  describe "scaled_dot_product_attention_with_mask/5" do
    test "additive mask (padding-style) matches defn fallback" do
      shape = {1, 2, 5, 8}
      q_len = elem(shape, 2)
      k_len = elem(shape, 2)

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend)

      # Pad the last key token (mask it out).
      mask_1d =
        Nx.tensor([0.0, 0.0, 0.0, 0.0, -1.0e9], type: :f32, backend: Nx.BinaryBackend)

      ref_mask = Nx.reshape(mask_1d, {1, 1, 1, k_len}) |> Nx.broadcast({1, 1, q_len, k_len})

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)
      mask = Nx.backend_copy(ref_mask, Emily.Backend)

      fun = fn q, k, v, mask ->
        Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, mask)
      end

      expected =
        Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_q, ref_k, ref_v, ref_mask)

      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v, mask)

      assert_close(fused, expected, tol: @f32_tol)
    end

    test "bf16 path within loose tolerance" do
      shape = {1, 4, 8, 16}

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend, :bf16)

      q_len = elem(shape, 2)
      k_len = elem(shape, 2)

      # Trivial all-zero (no-op) additive mask to exercise the
      # array-mask code path without altering the logits.
      ref_mask = Nx.broadcast(Nx.tensor(0.0, type: :bf16), {1, 1, q_len, k_len})

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)
      mask = Nx.backend_copy(ref_mask, Emily.Backend)

      fun = fn q, k, v, mask ->
        Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, mask)
      end

      expected =
        Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_q, ref_k, ref_v, ref_mask)

      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v, mask)

      assert_close(fused, expected, tol: @bf16_tol)
    end
  end

  describe "scaled_dot_product_attention/4 with :sinks" do
    test "sinks = neg_infinity reproduces the no-sinks result" do
      # Regression: when sinks = -inf the `exp(sinks - row_max)`
      # contribution vanishes and the denominator matches vanilla
      # softmax. Both paths (with/without sinks) must agree.
      shape = {1, 2, 4, 8}

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend)
      heads = elem(shape, 1)

      ref_sinks =
        Nx.Constants.neg_infinity(:f32)
        |> Nx.broadcast({heads})
        |> Nx.backend_copy(Nx.BinaryBackend)

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)
      sinks = Nx.backend_copy(ref_sinks, Emily.Backend)

      no_sinks_fun = fn q, k, v ->
        Emily.Fast.scaled_dot_product_attention(q, k, v)
      end

      sinks_fun = fn q, k, v, s ->
        Emily.Fast.scaled_dot_product_attention(q, k, v, sinks: s)
      end

      no_sinks = Nx.Defn.jit(no_sinks_fun, compiler: Emily.Compiler).(q, k, v)
      with_neg_inf = Nx.Defn.jit(sinks_fun, compiler: Emily.Compiler).(q, k, v, sinks)

      assert_close(with_neg_inf, no_sinks, tol: @f32_tol)
    end

    test "finite sinks shift probabilities by expected factor (hand-computed)" do
      # Shape: {batch=1, heads=1, seq_q=1, head_dim=2} / {..., seq_k=2, ...}.
      # Q dots K₀=[1,0] with cos=1, K₁=[0,1] with cos=0, so
      #   logits = [scale*1, scale*0] = [scale, 0].
      # With sink s=0:
      #   denom  = exp(scale) + exp(0) + exp(0) = exp(scale) + 2
      #   p₀     = exp(scale) / denom  (for V₀=[10,0])
      #   p₁     = 1          / denom  (for V₁=[20,0])
      #   out[0] = 10 * p₀ + 20 * p₁
      q = Nx.tensor([[[[1.0, 0.0]]]], type: :f32, backend: Emily.Backend)
      k = Nx.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], type: :f32, backend: Emily.Backend)
      v = Nx.tensor([[[[10.0, 0.0], [20.0, 0.0]]]], type: :f32, backend: Emily.Backend)
      sinks = Nx.tensor([0.0], type: :f32, backend: Emily.Backend)

      scale = 1.0 / :math.sqrt(2)

      fun = fn q, k, v, s ->
        Emily.Fast.scaled_dot_product_attention(q, k, v, sinks: s)
      end

      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v, sinks)

      e_scale = :math.exp(scale)
      denom = e_scale + 2.0
      p0 = e_scale / denom
      p1 = 1.0 / denom
      expected0 = 10.0 * p0 + 20.0 * p1

      assert_in_delta Nx.to_number(fused[0][0][0][0]), expected0, @f32_tol
      assert_in_delta Nx.to_number(fused[0][0][0][1]), 0.0, @f32_tol
    end

    test "fused matches defn fallback on a medium case (no mask)" do
      shape = {2, 4, 6, 16}

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend)
      heads = elem(shape, 1)

      # Small finite sinks so they actually influence the softmax.
      ref_sinks =
        Nx.iota({heads}, type: :f32, backend: Nx.BinaryBackend)
        |> Nx.multiply(0.25)
        |> Nx.subtract(0.5)

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)
      sinks = Nx.backend_copy(ref_sinks, Emily.Backend)

      fun = fn q, k, v, s ->
        Emily.Fast.scaled_dot_product_attention(q, k, v, sinks: s)
      end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_q, ref_k, ref_v, ref_sinks)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v, sinks)

      assert_close(fused, expected, tol: @f32_tol)
    end
  end

  describe "scaled_dot_product_attention_with_mask/5 with :sinks" do
    test "fused matches defn fallback with additive mask + sinks" do
      shape = {1, 2, 5, 8}
      q_len = elem(shape, 2)
      k_len = elem(shape, 2)
      heads = elem(shape, 1)

      {ref_q, ref_k, ref_v} = qkv_tensors(shape, Nx.BinaryBackend)

      mask_1d =
        Nx.tensor([0.0, 0.0, 0.0, 0.0, -1.0e9], type: :f32, backend: Nx.BinaryBackend)

      ref_mask = Nx.reshape(mask_1d, {1, 1, 1, k_len}) |> Nx.broadcast({1, 1, q_len, k_len})
      ref_sinks = Nx.tensor([-0.25, 0.25], type: :f32, backend: Nx.BinaryBackend)
      _ = heads

      q = Nx.backend_copy(ref_q, Emily.Backend)
      k = Nx.backend_copy(ref_k, Emily.Backend)
      v = Nx.backend_copy(ref_v, Emily.Backend)
      mask = Nx.backend_copy(ref_mask, Emily.Backend)
      sinks = Nx.backend_copy(ref_sinks, Emily.Backend)

      fun = fn q, k, v, m, s ->
        Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, m, sinks: s)
      end

      expected =
        Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_q, ref_k, ref_v, ref_mask, ref_sinks)

      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(q, k, v, mask, sinks)

      assert_close(fused, expected, tol: @f32_tol)
    end
  end
end
