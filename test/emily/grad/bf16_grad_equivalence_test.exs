defmodule Emily.Grad.Bf16EquivalenceTest do
  @moduledoc """
  bf16 gradient equivalence tests (M16).

  For each function in the grad zoo, casts the fixed inputs to bf16 and
  asserts that `grad` computed under `Emily.Compiler` in bf16 matches the
  exact gradient at those inputs: the same grad computed under
  `Nx.Defn.Evaluator` on `Nx.BinaryBackend` with the bf16 inputs widened
  (losslessly) to f32.

  bf16 has ~3 decimal digits of precision, so tolerances are 1e-2
  (matching `BackendGenerators.tol_for({:bf, _})`).

  ## Why the oracle isn't BinaryBackend bf16

  Since Nx 1.0 `Nx.Defn.grad` seeds the backward pass with a constant of
  the output's float type rather than f32, so a bf16 function's whole
  backward pass runs in bf16. `Nx.BinaryBackend` stores bf16 by truncating
  the f32 bits (round toward zero), and over a long backward pass
  (`grad_attention`) that bias drifts 2-3 bf16 ulps below the exact
  gradient, past the tolerance on its own. MLX rounds to nearest-even and
  stays within ~1 ulp. Comparing against the exact gradient checks
  Emily's bf16 path without inheriting the reference's rounding bias.

  Both results are cast to f32 before comparison.
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3, to_emily: 1]
  import Emily.GradZoo

  @bf16_tol 1.0e-2

  for name <- Emily.GradZoo.all_functions() do
    test "#{name} — bf16 grad matches the exact gradient" do
      name = unquote(name)
      fun = grad_function(name)
      inputs = fixed_inputs_bf16(name)

      emily_inputs = Enum.map(inputs, &to_emily/1)

      emily =
        Nx.Defn.jit_apply(fun, emily_inputs, compiler: Emily.Compiler)
        |> Nx.as_type({:f, 32})

      exact_inputs =
        Enum.map(inputs, fn
          %Nx.Tensor{type: {:bf, 16}} = t -> Nx.as_type(t, {:f, 32})
          t -> t
        end)

      ref =
        Nx.Defn.jit_apply(fun, exact_inputs, compiler: Nx.Defn.Evaluator)
        |> Nx.as_type({:f, 32})

      assert_close(emily, ref, tol: @bf16_tol)
    end
  end
end
