defmodule Emily.DebugFlagsTest do
  @moduledoc """
  M22 — Compile-time debug flag coverage.

    * Helpers raise on OOB / NaN when called directly.
    * Production OOB path is silent when `:debug_bounds_check` is off
      (negative control — confirms the failure class M22 addresses).
    * Fixture module proves the gate→helper composition fires when
      the compile-time flag is `true`.
    * Zero-cost verification — when the production flags are off,
      `Emily.Backend.beam` contains no call references to
      `DebugHelpers.check_bounds!` / `check_nan_inf!`.
  """

  use ExUnit.Case, async: true

  import Emily.TensorHelpers

  alias Emily.Backend.DebugHelpers
  alias Emily.DebugFixture

  # Captured at module compile time to pin the config the production
  # module was compiled against. `Application.compile_env/3` can only
  # be invoked from a module body, not a test function.
  @prod_bounds_flag Application.compile_env(:emily, :debug_bounds_check, false)
  @prod_naninf_flag Application.compile_env(:emily, :debug_detect_nan_inf, false)

  defp emily_ref(%Nx.Tensor{data: %Emily.Backend{ref: r}}), do: r

  describe "DebugHelpers.check_bounds!/5" do
    test "raises on positive out-of-range index" do
      idx = s32([0, 7, 2], [3])

      assert_raise ArgumentError, ~r/gather: index 7 on axis 0 out of range \(dim=5\)/, fn ->
        DebugHelpers.check_bounds!(:gather, {5, 4}, [idx], [0], worker())
      end
    end

    test "raises on negative index" do
      idx = s32([0, -1, 2], [3])

      assert_raise ArgumentError, ~r/take: index -1 on axis 0 is negative/, fn ->
        DebugHelpers.check_bounds!(:take, {5}, [idx], [0], worker())
      end
    end

    test "returns :ok on in-range indices" do
      idx = s32([0, 4, 2], [3])
      assert :ok = DebugHelpers.check_bounds!(:gather, {5}, [idx], [0], worker())
    end

    test "carries scatter op atom through to error message" do
      idx = s32([10], [1])

      assert_raise ArgumentError, ~r/indexed_add: index 10/, fn ->
        DebugHelpers.check_bounds!(:indexed_add, {3}, [idx], [0], worker())
      end
    end
  end

  describe "DebugHelpers.check_nan_inf!/3" do
    test "raises on NaN" do
      t = Nx.tensor([0.0, :nan, 2.0], type: {:f, 32}, backend: Emily.Backend)

      assert_raise ArgumentError, ~r/matmul: produced NaN or Inf/, fn ->
        DebugHelpers.check_nan_inf!(:matmul, emily_ref(t), worker())
      end
    end

    test "raises on positive infinity" do
      t = Nx.tensor([:infinity, 1.0], type: {:f, 32}, backend: Emily.Backend)

      assert_raise ArgumentError, ~r/fast_layer_norm: produced NaN or Inf/, fn ->
        DebugHelpers.check_nan_inf!(:fast_layer_norm, emily_ref(t), worker())
      end
    end

    test "returns :ok on finite tensor" do
      t = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}, backend: Emily.Backend)
      assert :ok = DebugHelpers.check_nan_inf!(:matmul, emily_ref(t), worker())
    end
  end

  describe "production flag off — silent OOB (negative control)" do
    test "Nx.take with out-of-range index does not raise" do
      t = Nx.tensor([10.0, 20.0, 30.0], type: {:f, 32}, backend: Emily.Backend)
      idx = Nx.tensor([0, 99], type: {:s, 32}, backend: Emily.Backend)

      # The MLX path returns whatever is at offset 99 — garbage or
      # zero. What matters for the negative control is that no
      # exception is raised with the flag off (the failure class M22
      # is designed to catch).
      result = Nx.take(t, idx)
      assert Nx.shape(result) == {2}
    end
  end

  describe "fixture: gate → helper composition" do
    test "bounds fixture raises with fixture flag on" do
      idx = s32([0, 99], [2])

      assert_raise ArgumentError, ~r/fixture_gather: index 99/, fn ->
        DebugFixture.bounds(:fixture_gather, {3}, [idx], [0], worker())
      end
    end

    test "bounds fixture returns :ok for in-range indices" do
      idx = s32([0, 2], [2])
      assert :ok = DebugFixture.bounds(:fixture_gather, {3}, [idx], [0], worker())
    end

    test "nan fixture raises with fixture flag on" do
      t = Nx.tensor([:nan, 1.0], type: {:f, 32}, backend: Emily.Backend)

      assert_raise ArgumentError, ~r/fixture_matmul: produced NaN or Inf/, fn ->
        DebugFixture.nan_inf(:fixture_matmul, emily_ref(t), worker())
      end
    end

    test "nan fixture returns :ok for finite tensor" do
      t = Nx.tensor([1.0, 2.0], type: {:f, 32}, backend: Emily.Backend)
      assert :ok = DebugFixture.nan_inf(:fixture_matmul, emily_ref(t), worker())
    end
  end

  describe "zero-cost verification" do
    # When a future refactor accidentally wires the helper
    # unconditionally, the MFA reference will land in the bytecode
    # and this test will catch it. We inspect the disassembled BEAM
    # (post-optimization) rather than the `:abstract_code` chunk
    # (pre-optimization) so compile-time-folded branches are gone.

    setup do
      {:beam_file, _mod, _exports, _attrs, _info, funs} =
        :beam_disasm.file(:code.which(Emily.Backend))

      {:ok, blob: :erlang.term_to_binary(funs)}
    end

    test "bounds-check helper reference is eliminated when flag is off", %{blob: blob} do
      refute @prod_bounds_flag
      refute blob =~ "check_bounds!"
    end

    test "nan/inf helper reference is eliminated when flag is off", %{blob: blob} do
      refute @prod_naninf_flag
      refute blob =~ "check_nan_inf!"
    end

    test "DebugHelpers module is not referenced at all when flags are off", %{blob: blob} do
      refute @prod_bounds_flag
      refute @prod_naninf_flag
      refute blob =~ "DebugHelpers"
    end
  end
end
