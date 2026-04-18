defmodule Emily.DebugFixture do
  @moduledoc false
  # Mirrors the compile-time gate pattern used in `Emily.Backend` so
  # tests can exercise the gate→helper composition end-to-end while the
  # production flags stay default-false. Fixture flags are set to `true`
  # in `config/test.exs`.

  alias Emily.Backend.DebugHelpers

  @flag_bounds Application.compile_env(:emily, :test_fixture_debug_bounds_check, false)
  @flag_nan Application.compile_env(:emily, :test_fixture_debug_detect_nan_inf, false)

  @spec bounds(atom(), tuple(), [reference()], [integer()], reference()) :: :ok
  def bounds(op, input_shape, idx_refs, axes, w) do
    if @flag_bounds, do: DebugHelpers.check_bounds!(op, input_shape, idx_refs, axes, w)
    :ok
  end

  @spec nan_inf(atom(), reference(), reference()) :: :ok
  def nan_inf(op, result_ref, w) do
    if @flag_nan, do: DebugHelpers.check_nan_inf!(op, result_ref, w)
    :ok
  end
end
