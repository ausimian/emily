defmodule Emily.CompilerNativeConfigTest do
  @moduledoc """
  `config :emily, :native` sets the default for the compiler's `:native`
  option when a call passes none; the shipped default (no config) is
  `true`. A per-call `native:` always wins over the app env. Mutates the
  application env, so `async: false`.

  Probe: `Nx.reduce/3` with a BEAM reducer can never lower to the single-NIF
  replay (the reducer would have to run on the host mid-graph). Under
  `native_fallback: :raise` it therefore raises *iff* native is enabled —
  a clean signal for how `:native` resolved, without inspecting the
  compiled program.
  """
  use ExUnit.Case, async: false

  defp unsupported_fun, do: fn x -> Nx.reduce(x, 0.0, fn a, b -> Nx.add(a, b) end) end

  defp t(data), do: Nx.tensor(data, backend: Emily.Backend)

  setup do
    prev = Application.fetch_env(:emily, :native)

    on_exit(fn ->
      case prev do
        {:ok, v} -> Application.put_env(:emily, :native, v)
        :error -> Application.delete_env(:emily, :native)
      end
    end)

    :ok
  end

  test "with no config the default is native (no per-call option)" do
    Application.delete_env(:emily, :native)

    assert_raise ArgumentError, ~r/reduce/, fn ->
      Nx.Defn.jit(unsupported_fun(), compiler: Emily.Compiler, native_fallback: :raise).(
        t([1.0, 2.0])
      )
    end
  end

  test "config :emily, native: true enables native (no per-call option)" do
    Application.put_env(:emily, :native, true)

    assert_raise ArgumentError, ~r/reduce/, fn ->
      Nx.Defn.jit(unsupported_fun(), compiler: Emily.Compiler, native_fallback: :raise).(
        t([1.0, 2.0])
      )
    end
  end

  test "config :emily, native: false routes through the evaluator" do
    Application.put_env(:emily, :native, false)

    out =
      Nx.Defn.jit(unsupported_fun(), compiler: Emily.Compiler, native_fallback: :raise).(
        t([3.0, 1.0, 2.0])
      )

    assert Nx.to_number(out) == 6.0
  end

  test "a per-call native: false overrides config :emily, native: true" do
    Application.put_env(:emily, :native, true)

    out =
      Nx.Defn.jit(unsupported_fun(),
        compiler: Emily.Compiler,
        native: false,
        native_fallback: :raise
      ).(t([3.0, 1.0, 2.0]))

    assert Nx.to_number(out) == 6.0
  end

  test "a non-boolean :native raises" do
    Application.put_env(:emily, :native, :yes)

    assert_raise ArgumentError, ~r/invalid :native/, fn ->
      Nx.Defn.jit(fn x -> Nx.add(x, 1.0) end, compiler: Emily.Compiler).(t([1.0]))
    end
  end
end
