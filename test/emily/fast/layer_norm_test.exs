defmodule Emily.Fast.LayerNormTest do
  @moduledoc """
  Tests for `Emily.Fast.layer_norm/4` covering fallback correctness,
  defn composability, and fused-kernel equivalence under Emily.
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3]

  @f32_tol 1.0e-5
  @bf16_tol 1.0e-2

  defp reference_layer_norm(x, weight, bias, eps) do
    orig_type = Nx.type(x)
    x_f32 = Nx.as_type(x, :f32)
    mean = Nx.mean(x_f32, axes: [-1], keep_axes: true)
    centered = Nx.subtract(x_f32, mean)
    var = Nx.mean(Nx.pow(centered, 2), axes: [-1], keep_axes: true)
    normalized = Nx.multiply(centered, Nx.rsqrt(Nx.add(var, eps)))

    normalized
    |> Nx.as_type(orig_type)
    |> Nx.multiply(weight)
    |> Nx.add(bias)
  end

  describe "fallback correctness (BinaryBackend)" do
    test "matches hand-rolled reference" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [-0.5, 0.25, 1.0, -2.0]], backend: Nx.BinaryBackend)
      w = Nx.tensor([0.5, 1.0, 1.5, 2.0], backend: Nx.BinaryBackend)
      b = Nx.tensor([0.1, -0.1, 0.2, -0.2], backend: Nx.BinaryBackend)

      fun = fn x, w, b -> Emily.Fast.layer_norm(x, w, b, eps: 1.0e-5) end
      got = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(x, w, b)
      expected = reference_layer_norm(x, w, b, 1.0e-5)

      assert_close(got, expected, tol: @f32_tol)
    end
  end

  describe "emily backend (fused path)" do
    setup do
      Nx.default_backend(Emily.Backend)
      :ok
    end

    test "matches BinaryBackend oracle for f32" do
      shape = {3, 5, 16}

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(100)
      ref_w = Nx.iota({16}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(1) |> Nx.divide(16)
      ref_b = Nx.iota({16}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(20)

      x = Nx.backend_copy(ref_x, Emily.Backend)
      w = Nx.backend_copy(ref_w, Emily.Backend)
      b = Nx.backend_copy(ref_b, Emily.Backend)

      fun = fn x, w, b -> Emily.Fast.layer_norm(x, w, b, eps: 1.0e-5) end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_w, ref_b)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, w, b)

      assert_close(fused, expected, tol: @f32_tol)
    end

    test "matches BinaryBackend oracle for bf16" do
      shape = {2, 8, 16}

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(50)
      ref_w = Nx.iota({16}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(1) |> Nx.divide(16)
      ref_b = Nx.iota({16}, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(20)

      [ref_x, ref_w, ref_b] = Enum.map([ref_x, ref_w, ref_b], &Nx.as_type(&1, :bf16))

      x = Nx.backend_copy(ref_x, Emily.Backend)
      w = Nx.backend_copy(ref_w, Emily.Backend)
      b = Nx.backend_copy(ref_b, Emily.Backend)

      fun = fn x, w, b -> Emily.Fast.layer_norm(x, w, b, eps: 1.0e-5) end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_w, ref_b)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, w, b)

      assert_close(fused, expected, tol: @bf16_tol)
    end
  end
end
