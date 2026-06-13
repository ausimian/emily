defmodule Emily.CompilerFallbackTest do
  @moduledoc """
  CM7 — graceful whole-defn fallback. When a `native: true` defn hits an
  op the Expr compiler can't lower, the default (`native_fallback: :eval`)
  routes the *whole* defn through `Nx.Defn.Evaluator` (each op then
  dispatches through `Emily.Backend`) and fires a
  `[:emily, :compiler, :fallback]` telemetry event — rather than raising.
  `native_fallback: :raise` restores the strict no-fallback behaviour the
  conformance gates rely on.

  These pass `:native_fallback` per call (the test suite default is
  `:raise`, set in `config/test.exs`), so no global state is mutated and
  the cases stay `async`.
  """
  use ExUnit.Case, async: true
  import Nx.Defn

  # `Nx.reduce` with an arbitrary BEAM reducer can never lower to the
  # single-NIF replay (the reducer would have to run on the host mid-graph),
  # so it is a *stable* example of an op the compiler routes to the
  # evaluator — unlike a specific primitive, which a later milestone may
  # teach the IR to lower. The evaluator runs it via `Emily.Backend`, so the
  # fallback result is correct (reduce-with-add == sum).
  defp unsupported_fun, do: fn x -> Nx.reduce(x, 0.0, fn a, b -> Nx.add(a, b) end) end

  # Fully supported by the IR — must still compile native, no fallback.
  defn(supported_fn(x), do: Nx.add(Nx.multiply(x, 2.0), 1.0))

  defp t(data), do: Nx.tensor(data, backend: Emily.Backend)

  # Named module-function handler (not an anonymous fn) so `:telemetry`
  # doesn't log its local-function performance note during the run.
  @doc false
  def forward_event(_event, meas, meta, {pid, ref}), do: send(pid, {ref, :event, meas, meta})

  defp attach(event, ref) do
    id = "cm7-#{inspect(ref)}"
    :telemetry.attach(id, event, &__MODULE__.forward_event/4, {self(), ref})
    on_exit(fn -> :telemetry.detach(id) end)
  end

  describe "native_fallback: :eval (graceful, the runtime default)" do
    test "an unsupported op falls back to the evaluator and matches it" do
      fun = unsupported_fun()
      x = t([3.0, 1.0, 2.0, 0.0])

      native =
        Nx.Defn.jit(fun, compiler: Emily.Compiler, native: true, native_fallback: :eval).(x)

      eval = Nx.Defn.jit(fun, compiler: Emily.Compiler).(x)

      assert %Emily.Backend{} = native.data
      assert Nx.to_binary(native) == Nx.to_binary(eval)
    end

    test "fires a [:emily, :compiler, :fallback] event naming the op" do
      ref = make_ref()
      attach([:emily, :compiler, :fallback], ref)

      Nx.Defn.jit(unsupported_fun(),
        compiler: Emily.Compiler,
        native: true,
        native_fallback: :eval
      ).(t([2.0, 1.0]))

      assert_receive {^ref, :event, %{count: 1}, %{reason: reason}}
      assert reason =~ "reduce"
    end

    test "a fully supported defn compiles native rather than falling back" do
      # Asserted under `:raise` (not by checking the absence of a global
      # telemetry event, which would race with concurrent async modules):
      # with no fallback available, success proves the defn lowered fully
      # native instead of silently degrading to the evaluator.
      x = t([1.0, 2.0, 3.0])

      out =
        Nx.Defn.jit(&supported_fn/1,
          compiler: Emily.Compiler,
          native: true,
          native_fallback: :raise
        ).(x)

      assert %Emily.Backend{} = out.data

      assert Nx.to_binary(out) ==
               Nx.to_binary(Nx.Defn.jit(&supported_fn/1, compiler: Emily.Compiler).(x))
    end
  end

  describe "native_fallback: :raise (strict, the conformance-gate mode)" do
    test "an unsupported op raises rather than falling back" do
      assert_raise ArgumentError, ~r/reduce/, fn ->
        Nx.Defn.jit(unsupported_fun(),
          compiler: Emily.Compiler,
          native: true,
          native_fallback: :raise
        ).(t([1.0, 2.0]))
      end
    end
  end
end
