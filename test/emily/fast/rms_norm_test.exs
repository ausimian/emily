defmodule Emily.Fast.RMSNormTest do
  @moduledoc """
  Tests for `Emily.Fast.rms_norm/3`.

  Three axes of coverage:

    * **Fallback correctness** — on `Nx.BinaryBackend` the defn
      fallback must match a hand-computed reference.
    * **Defn composability** — inside a `defn` jitted with
      `Emily.Compiler`, the helper returns a sensible result and its
      shape/type propagate.
    * **Fused vs composed equivalence** — the fused MLX kernel under
      Emily must agree with the BinaryBackend fallback within a
      dtype-aware tolerance (MLX reorders ops inside the fused rms_norm
      so bit-match isn't expected).
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3]

  @f32_tol 1.0e-5
  @bf16_tol 1.0e-2

  defp reference_rms_norm(x, weight, eps) do
    # Textbook RMSNorm with the f32 upcast recipe.
    orig_type = Nx.type(x)
    x_f32 = Nx.as_type(x, :f32)
    var = Nx.mean(Nx.pow(x_f32, 2), axes: [-1], keep_axes: true)
    normalized = Nx.multiply(x_f32, Nx.rsqrt(Nx.add(var, eps)))

    normalized
    |> Nx.as_type(orig_type)
    |> Nx.multiply(weight)
  end

  describe "fallback correctness (BinaryBackend)" do
    test "matches hand-rolled reference" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [-0.5, 0.25, 1.0, -2.0]], backend: Nx.BinaryBackend)
      weight = Nx.tensor([0.5, 1.0, 1.5, 2.0], backend: Nx.BinaryBackend)

      fun = fn x, w -> Emily.Fast.rms_norm(x, w, eps: 1.0e-6) end
      got = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(x, weight)
      expected = reference_rms_norm(x, weight, 1.0e-6)

      assert_close(got, expected, tol: @f32_tol)
    end
  end

  describe "emily backend (fused path)" do
    setup do
      Nx.default_backend(Emily.Backend)
      :ok
    end

    test "matches BinaryBackend oracle for f32" do
      shape = {2, 8, 32}

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(100)
      ref_w = Nx.iota({32}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(1) |> Nx.divide(32)

      x = Nx.backend_copy(ref_x, Emily.Backend)
      w = Nx.backend_copy(ref_w, Emily.Backend)

      fun = fn x, w -> Emily.Fast.rms_norm(x, w, eps: 1.0e-6) end

      expected =
        Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_w)

      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, w)

      assert_close(fused, expected, tol: @f32_tol)
    end

    test "matches BinaryBackend oracle for bf16" do
      shape = {1, 4, 16}

      ref_x = Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.divide(50)
      ref_w = Nx.iota({16}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(1) |> Nx.divide(16)

      ref_x_bf = Nx.as_type(ref_x, :bf16)
      ref_w_bf = Nx.as_type(ref_w, :bf16)

      x = Nx.backend_copy(ref_x_bf, Emily.Backend)
      w = Nx.backend_copy(ref_w_bf, Emily.Backend)

      fun = fn x, w -> Emily.Fast.rms_norm(x, w, eps: 1.0e-6) end

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x_bf, ref_w_bf)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, w)

      assert_close(fused, expected, tol: @bf16_tol)
    end

    test "preserves input dtype" do
      x = Nx.broadcast(Nx.tensor(1.0, type: :f32), {2, 4})
      w = Nx.broadcast(Nx.tensor(1.0, type: :f32), {4})

      fused = Nx.Defn.jit(&Emily.Fast.rms_norm/3, compiler: Emily.Compiler).(x, w, eps: 1.0e-6)
      assert Nx.type(fused) == {:f, 32}
      assert Nx.shape(fused) == {2, 4}
    end

    test "composes with surrounding ops in defn" do
      fun = fn x, w ->
        x
        |> Nx.multiply(2.0)
        |> Emily.Fast.rms_norm(w, eps: 1.0e-6)
        |> Nx.add(1.0)
      end

      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]], backend: Emily.Backend)
      w = Nx.tensor([0.5, 1.0, 1.5, 2.0], backend: Emily.Backend)

      ref_x = Nx.backend_copy(x, Nx.BinaryBackend)
      ref_w = Nx.backend_copy(w, Nx.BinaryBackend)

      expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(ref_x, ref_w)
      fused = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x, w)
      assert_close(fused, expected, tol: @f32_tol)
    end
  end
end
