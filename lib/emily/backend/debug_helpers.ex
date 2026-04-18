defmodule Emily.Backend.DebugHelpers do
  @moduledoc false
  # Assertion bodies for the M22 compile-time debug flags. Always
  # compiled; `Emily.Backend` only calls into here from `if @flag` gates
  # that fold to nothing when the attribute is the literal `false`, so
  # the reference to this module never lands in `Emily.Backend.beam`
  # under the default-off build.

  alias Emily.Native

  @doc false
  @spec check_bounds!(atom(), tuple(), [reference()], [integer()], reference()) :: :ok
  def check_bounds!(op, input_shape, idx_refs, axes, w) do
    axes
    |> Enum.zip(idx_refs)
    |> Enum.each(fn {axis, idx_ref} ->
      dim = elem(input_shape, axis)
      red_axes = Enum.to_list(0..(length(Native.shape(idx_ref)) - 1))

      max_ref = Native.astype(w, Native.max(w, idx_ref, red_axes, false), {:s, 32})
      min_ref = Native.astype(w, Native.min(w, idx_ref, red_axes, false), {:s, 32})
      <<max_i::signed-integer-32-native>> = Native.to_binary(w, max_ref)
      <<min_i::signed-integer-32-native>> = Native.to_binary(w, min_ref)

      cond do
        min_i < 0 ->
          raise ArgumentError,
                "#{op}: index #{min_i} on axis #{axis} is negative (dim=#{dim})"

        max_i >= dim ->
          raise ArgumentError,
                "#{op}: index #{max_i} on axis #{axis} out of range (dim=#{dim})"

        true ->
          :ok
      end
    end)
  end

  @doc false
  @spec check_nan_inf!(atom(), reference(), reference()) :: :ok
  def check_nan_inf!(op, result_ref, w) do
    bad = Native.logical_or(w, Native.isnan(w, result_ref), Native.isinf(w, result_ref))
    axes = Enum.to_list(0..(length(Native.shape(bad)) - 1))
    scalar = Native.any(w, bad, axes, false)
    <<flag::unsigned-integer-8>> = Native.to_binary(w, scalar)
    if flag == 1, do: raise(ArgumentError, "#{op}: produced NaN or Inf")
    :ok
  end
end
