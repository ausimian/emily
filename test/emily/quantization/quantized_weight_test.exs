defmodule Emily.QuantizedWeightTest do
  @moduledoc """
  Unit and property tests for `Emily.QuantizedWeight`. Covers:

    * `from_dense/2` shape/dtype metadata and validation
    * `to_dense/1` inverse correctness within int4/int8 tolerance
    * `Nx.Container` semantics — tensor fields traverse, scalar metadata
      survives the traversal via `keep:`
    * `Nx.backend_transfer/2` on the whole container
  """

  use ExUnit.Case, async: true
  use ExUnitProperties
  doctest Emily.QuantizedWeight

  alias Emily.QuantizedWeight

  import Emily.BackendGenerators

  describe "from_dense/2" do
    test "default opts produce a group_size=64, bits=4 packed weight" do
      w =
        Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(512)
        |> Nx.subtract(0.25)

      qw = QuantizedWeight.from_dense(w)

      assert %QuantizedWeight{group_size: 64, bits: 4, transpose: true} = qw
      assert Nx.type(qw.value) == {:u, 32}
      # 128 nibbles → 128 / 8 = 16 u32 per row.
      assert Nx.shape(qw.value) == {2, 16}
      assert Nx.type(qw.scales) == {:f, 32}
      # 128 / 64 = 2 groups per row.
      assert Nx.shape(qw.scales) == {2, 2}
      assert Nx.shape(qw.biases) == {2, 2}
    end

    test "accepts overrides for group_size, bits, transpose" do
      w = Nx.iota({1, 64}, backend: Emily.Backend, type: :f32)
      qw = QuantizedWeight.from_dense(w, group_size: 32, bits: 8, transpose: false)

      assert qw.group_size == 32
      assert qw.bits == 8
      assert qw.transpose == false
      # 64 bytes packed into u32 → 16 values per row.
      assert Nx.shape(qw.value) == {1, 16}
      # 64 / 32 = 2 groups per row.
      assert Nx.shape(qw.scales) == {1, 2}
    end

    test "raises on indivisible last axis" do
      w = Nx.iota({1, 30}, backend: Emily.Backend, type: :f32)

      assert_raise ArgumentError, ~r/divisible by :group_size/, fn ->
        QuantizedWeight.from_dense(w)
      end
    end

    test "raises on unsupported bits" do
      w = Nx.iota({1, 64}, backend: Emily.Backend, type: :f32)

      assert_raise ArgumentError, ~r/:bits must be one of/, fn ->
        QuantizedWeight.from_dense(w, bits: 5)
      end
    end

    test "raises on unsupported dtype" do
      w = Nx.iota({1, 64}, backend: Emily.Backend, type: :s32)

      assert_raise ArgumentError, ~r/from_dense\/2 requires/, fn ->
        QuantizedWeight.from_dense(w)
      end
    end

    test "raises on rank-1 input (MLX requires rank ≥ 2)" do
      w = Nx.iota({128}, backend: Emily.Backend, type: :f32)

      assert_raise ArgumentError, ~r/rank/, fn ->
        QuantizedWeight.from_dense(w)
      end
    end
  end

  describe "to_dense/1" do
    property "from_dense |> to_dense recovers input within int4 tolerance" do
      check all(
              rows <- StreamData.integer(1..4),
              groups <- StreamData.integer(1..3),
              max_runs: 12
            ) do
        cols = groups * 64

        # Values spread across [-1, 1] so the per-group dynamic range is
        # non-trivial; int4 step across a 2.0-wide range is ~2/15 ≈ 0.13.
        w =
          Nx.iota({rows, cols}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(rows * cols / 2)
          |> Nx.subtract(1.0)

        qw = QuantizedWeight.from_dense(w)
        recovered = QuantizedWeight.to_dense(qw)

        assert Nx.shape(recovered) == Nx.shape(w)
        assert Nx.type(recovered) == Nx.type(w)

        # Conservative bound: int4 step across [-1, 1] per group is
        # ≤ 2/15; observed max absolute error on these fixtures is ~0.07.
        assert_close(recovered, w, tol: 0.15)
      end
    end

    test "bits=8 is tighter than bits=4" do
      values = for i <- 0..127, do: (i - 64) / 128.0
      w = Nx.tensor(values, backend: Emily.Backend, type: :f32) |> Nx.reshape({1, 128})

      q4 = w |> QuantizedWeight.from_dense(bits: 4) |> QuantizedWeight.to_dense()
      q8 = w |> QuantizedWeight.from_dense(bits: 8) |> QuantizedWeight.to_dense()

      err4 = Nx.subtract(q4, w) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      err8 = Nx.subtract(q8, w) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert err8 < err4
    end
  end

  describe "Nx.Container semantics" do
    test "backend_transfer walks the three tensor fields and keeps metadata" do
      w = Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
      qw = QuantizedWeight.from_dense(w)

      transferred = Nx.backend_transfer(qw, Nx.BinaryBackend)

      # Scalar metadata survives the traversal (keep: in the derive).
      assert transferred.group_size == 64
      assert transferred.bits == 4
      assert transferred.transpose == true
      assert transferred.mode == "affine"

      # Tensor fields were walked — each now lives on BinaryBackend.
      assert Nx.backend_transfer(transferred.value, Nx.BinaryBackend) == transferred.value
      assert Nx.backend_transfer(transferred.scales, Nx.BinaryBackend) == transferred.scales
    end
  end

  describe "microscaled modes (mxfp4 / mxfp8 / nvfp4)" do
    # MLX's `fp_quantize` path pins {group_size, bits} per mode. See
    # `vendor/mlx/mlx/ops.cpp:4808-4823` for the canonical constraints.
    @cases [
      {"mxfp4", 32, 4, 0.3},
      {"mxfp8", 32, 8, 0.05},
      {"nvfp4", 16, 4, 0.3}
    ]

    for {mode, group_size, bits, tol} <- @cases do
      test "from_dense(#{mode}) → to_dense round-trips within #{tol}" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)
        tol = unquote(tol)

        w =
          Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(256)
          |> Nx.subtract(0.25)

        qw = QuantizedWeight.from_dense(w, mode: mode, group_size: group_size, bits: bits)

        assert qw.mode == mode
        assert qw.group_size == group_size
        assert qw.bits == bits

        # Microscaled scales pack exponent bits into a u8 tensor.
        assert Nx.type(qw.scales) == {:u, 8}

        dense = QuantizedWeight.to_dense(qw)
        assert Nx.shape(dense) == Nx.shape(w)
        # MLX's fp_dequantize defaults to bfloat16 output when no
        # out_type is requested (see `validate_mode_with_type` in
        # `vendor/mlx/mlx/ops.cpp:4442-4446`). Cast before comparing.
        assert Nx.type(dense) == {:bf, 16}
        dense_f32 = Nx.as_type(dense, :f32)
        assert_close(dense_f32, w, tol: tol)
      end
    end

    test "mxfp4 rejects non-default group_size" do
      w = Nx.iota({1, 128}, backend: Emily.Backend, type: :f32)

      assert_raise ArgumentError, ~r/mxfp4.*group_size=32/, fn ->
        QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 64, bits: 4)
      end
    end

    test "mxfp8 rejects bits != 8" do
      w = Nx.iota({1, 128}, backend: Emily.Backend, type: :f32)

      assert_raise ArgumentError, ~r/mxfp8.*bits=8/, fn ->
        QuantizedWeight.from_dense(w, mode: "mxfp8", group_size: 32, bits: 4)
      end
    end

    test "unknown mode raises a clear error" do
      w = Nx.iota({1, 128}, backend: Emily.Backend, type: :f32)

      assert_raise ArgumentError, ~r/:mode must be one of/, fn ->
        QuantizedWeight.from_dense(w, mode: "int2", group_size: 32, bits: 4)
      end
    end

    test "backend_transfer keeps mode metadata for microscaled modes" do
      w = Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
      qw = QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 32, bits: 4)

      transferred = Nx.backend_transfer(qw, Nx.BinaryBackend)
      assert transferred.mode == "mxfp4"
      assert transferred.group_size == 32
      assert transferred.bits == 4
    end
  end
end
