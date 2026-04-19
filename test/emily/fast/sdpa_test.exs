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
end
