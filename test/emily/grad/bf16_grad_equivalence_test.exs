defmodule Emily.Grad.Bf16EquivalenceTest do
  @moduledoc """
  bf16 gradient equivalence tests (M16).

  For each function in the grad zoo, casts the fixed inputs to bf16 and
  asserts that `grad` computed under `Emily.Compiler` matches the same
  grad computed under `Nx.Defn.Evaluator` on `Nx.BinaryBackend` — both
  sides in bf16. The oracle is BinaryBackend's software bf16, not f32.

  bf16 has ~3 decimal digits of precision, so tolerances are 1e-2
  (matching `BackendGenerators.tol_for({:bf, _})`).

  ## Note on grad type promotion

  `Nx.Defn.grad` may promote the output type from bf16 to f32 in
  certain backward ops. Both the Emily and BinaryBackend results are
  cast to f32 before comparison to normalise the binary
  representation — the bf16-level tolerance still applies because
  the computation itself ran in bf16.
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3, to_emily: 1]
  import Emily.GradZoo

  @bf16_tol 1.0e-2

  for name <- Emily.GradZoo.all_functions() do
    test "#{name} — bf16 grad matches BinaryBackend bf16 oracle" do
      name = unquote(name)
      fun = grad_function(name)
      inputs = fixed_inputs_bf16(name)

      emily_inputs = Enum.map(inputs, &to_emily/1)

      emily =
        Nx.Defn.jit_apply(fun, emily_inputs, compiler: Emily.Compiler)
        |> Nx.as_type({:f, 32})

      ref =
        Nx.Defn.jit_apply(fun, inputs, compiler: Nx.Defn.Evaluator)
        |> Nx.as_type({:f, 32})

      assert_close(emily, ref, tol: @bf16_tol)
    end
  end
end
