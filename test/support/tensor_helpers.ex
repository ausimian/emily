defmodule Emily.TensorHelpers do
  @moduledoc """
  Shared helpers for Emily.Native NIF tests. Build tensors from Elixir
  lists, pull them back as lists, and compare floats with tolerance.
  """

  alias Emily.Native

  @doc "Build an f32 tensor from a numeric list and shape."
  def f32(list, shape) when is_list(list) do
    bin = for x <- list, into: <<>>, do: <<x * 1.0::float-32-native>>
    Native.from_binary(bin, shape, {:f, 32})
  end

  @doc "Build a rank-0 f32 tensor."
  def f32_scalar(x), do: f32([x], [])

  @doc "Read an f32 tensor back as a flat list of floats."
  def to_f32_list(tensor) do
    bin = Native.to_binary(tensor)
    for <<f::float-32-native <- bin>>, do: f
  end

  @doc "Build an s32 tensor from an integer list and shape."
  def s32(list, shape) when is_list(list) do
    bin = for x <- list, into: <<>>, do: <<x::signed-integer-32-native>>
    Native.from_binary(bin, shape, {:s, 32})
  end

  @doc "Read an s32 tensor back as a flat list of ints."
  def to_s32_list(tensor) do
    bin = Native.to_binary(tensor)
    for <<i::signed-integer-32-native <- bin>>, do: i
  end

  @doc "Build a pred (bool) tensor from a list of booleans and shape."
  def pred(list, shape) when is_list(list) do
    bin = for b <- list, into: <<>>, do: <<if(b, do: 1, else: 0)::8>>
    Native.from_binary(bin, shape, {:pred, 1})
  end

  @doc "Read a pred tensor back as a flat list of booleans."
  def to_pred_list(tensor) do
    bin = Native.to_binary(tensor)
    for <<b::unsigned-integer-8 <- bin>>, do: b == 1
  end

  @doc """
  Assert element-wise proximity with combined absolute + relative
  tolerance: `|a - e| <= tol * (1 + |e|)`.
  """
  def assert_close(actual, expected, tol \\ 1.0e-5)

  def assert_close(actual, expected, tol) when is_list(actual) and is_list(expected) do
    ExUnit.Assertions.assert(
      length(actual) == length(expected),
      "length mismatch: #{inspect(actual)} vs #{inspect(expected)}"
    )

    Enum.zip(actual, expected)
    |> Enum.each(fn {a, e} -> assert_close(a, e, tol) end)
  end

  def assert_close(actual, expected, tol)
      when is_number(actual) and is_number(expected) do
    if abs(actual - expected) <= tol + tol * abs(expected) do
      :ok
    else
      ExUnit.Assertions.flunk("expected #{expected}, got #{actual} (tol=#{tol})")
    end
  end
end
