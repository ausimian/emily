defmodule Emily.OpcodeParityTest do
  @moduledoc """
  Guards the one hand-maintained lockstep in the native compiler: the
  opcode wire values live in **two** places that must agree —
  `Emily.IR`'s `@opcodes` map (Elixir) and the `Opcode` enum +
  `kOpcodeCount` in `c_src/emily/opcodes.hpp` (C++). Nothing in the build
  enforces it: a mismatch compiles fine and only misbehaves at runtime
  (an instruction dispatches to the wrong MLX op, or `valid_opcode`
  rejects a real one).

  The check is value-based rather than name-based — both sides must be a
  contiguous `0..N-1` with N == `kOpcodeCount`. That catches every
  realistic drift (an opcode added to one side only, a forgotten
  `kOpcodeCount` bump, a duplicate or gapped number) without depending on
  the Elixir snake_case names matching the C++ PascalCase ones (they
  don't always: `negate`/`Negative`, `fast_rms_norm`/`FastRMSNorm`). A
  name/value *permutation* that keeps both contiguous would slip past
  here, but that produces wrong results and is caught by
  `Emily.CompilerEquivalenceTest`.
  """
  use ExUnit.Case, async: true

  alias Emily.IR

  @header Path.expand("../../c_src/emily/opcodes.hpp", __DIR__)

  # Parse `kOpcodeCount` and the `Opcode` enum's explicit `Name = N,`
  # values out of the C++ header.
  defp parse_header do
    src = File.read!(@header)

    [_, count] = Regex.run(~r/kOpcodeCount\s*=\s*(\d+)/, src)
    count = String.to_integer(count)

    [_, body] = Regex.run(~r/enum class Opcode\s*:\s*int64_t\s*\{(.*?)\};/s, src)

    values =
      ~r/^\s*[A-Za-z]\w*\s*=\s*(\d+)\s*,/m
      |> Regex.scan(body)
      |> Enum.map(fn [_, v] -> String.to_integer(v) end)

    %{count: count, enum_values: values}
  end

  test "C++ Opcode enum is contiguous 0..N-1 and matches kOpcodeCount" do
    %{count: count, enum_values: values} = parse_header()

    assert length(values) == count,
           "opcodes.hpp has #{length(values)} enum entries but kOpcodeCount is #{count} — " <>
             "bump kOpcodeCount when adding an opcode"

    assert Enum.sort(values) == Enum.to_list(0..(count - 1)),
           "Opcode enum values are not a unique, gap-free 0..#{count - 1}"
  end

  test "Emily.IR @opcodes stays in lockstep with the C++ enum" do
    %{count: count} = parse_header()
    opcodes = IR.opcodes()

    assert map_size(opcodes) == count,
           "Emily.IR has #{map_size(opcodes)} opcodes but c_src/emily/opcodes.hpp kOpcodeCount " <>
             "is #{count} — the two must be updated together"

    assert opcodes |> Map.values() |> Enum.sort() == Enum.to_list(0..(count - 1)),
           "Emily.IR @opcodes values are not a unique, gap-free 0..#{count - 1}"
  end
end
