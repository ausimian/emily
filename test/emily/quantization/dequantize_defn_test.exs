defmodule Emily.Quantization.DequantizeDefnTest do
  @moduledoc """
  Equality tests for `Emily.Quantization.dequantize_defn/1` against
  `QuantizedWeight.to_dense/1` (the Native path). Covers every
  `bits ∈ {2, 3, 4, 6, 8}` × `group_size ∈ {32, 64, 128}` combo,
  plus a rank-3 case and defn composability. Microscaled modes are
  rejected by the defn path; negative test below.
  """

  use ExUnit.Case, async: true
  doctest Emily.Quantization

  alias Emily.Quantization
  alias Emily.QuantizedWeight

  import Emily.BackendGenerators, only: [assert_close: 3]

  @bits [2, 3, 4, 6, 8]
  @group_sizes [32, 64, 128]

  describe "dequantize_defn/1 — equality with QuantizedWeight.to_dense/1" do
    for bits <- @bits, group_size <- @group_sizes do
      test "bits=#{bits}, group_size=#{group_size}" do
        bits = unquote(bits)
        group_size = unquote(group_size)

        out_feat = 4
        in_feat = group_size * 3

        # Small-magnitude values so int4 quantization error doesn't
        # swamp the comparison — which is against the *dequantized*
        # oracle, not the original dense, so the quantization error
        # cancels anyway.
        w =
          Nx.iota({out_feat, in_feat}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(out_feat * in_feat / 2)
          |> Nx.subtract(1.0)

        qw = QuantizedWeight.from_dense(w, group_size: group_size, bits: bits)

        actual = Quantization.dequantize_defn(qw)
        expected = QuantizedWeight.to_dense(qw)

        assert Nx.shape(actual) == Nx.shape(expected)
        assert Nx.type(actual) == Nx.type(expected)
        # f32 reconstruction is mathematically identical on both paths;
        # 1e-6 is a tight safety tolerance.
        assert_close(actual, expected, tol: 1.0e-6)
      end
    end

    for bits <- @bits do
      test "rank ≥ 3 input (batch of matrices), bits=#{bits}" do
        bits = unquote(bits)
        # QuantizedWeight itself requires rank ≥ 2, but the last-axis
        # quantization scheme means any leading axes are carried through
        # unchanged. Verify that the defn path handles rank-3 — including
        # the cross-u32 path (bits ∈ {3, 6}), which dispatches through
        # `Nx.take(_, axis: -1)` on the packed buffer (a distinct shape
        # from the integral path's broadcast-shift).
        w =
          Nx.iota({2, 3, 128}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(2 * 3 * 128 / 2)
          |> Nx.subtract(1.0)

        qw = QuantizedWeight.from_dense(w, group_size: 64, bits: bits)

        actual = Quantization.dequantize_defn(qw)
        expected = QuantizedWeight.to_dense(qw)

        assert Nx.shape(actual) == {2, 3, 128}
        assert_close(actual, expected, tol: 1.0e-6)
      end
    end

    for bits <- @bits do
      test "bits=#{bits} works after transfer to Nx.BinaryBackend" do
        bits = unquote(bits)

        w =
          Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(256.0)
          |> Nx.subtract(0.25)

        qw = QuantizedWeight.from_dense(w, group_size: 64, bits: bits)
        expected = QuantizedWeight.to_dense(qw) |> Nx.backend_transfer(Nx.BinaryBackend)

        actual =
          qw
          |> Nx.backend_transfer(Nx.BinaryBackend)
          |> Quantization.dequantize_defn()

        assert Nx.shape(actual) == Nx.shape(expected)
        # Confirm dequantize_defn stayed on BinaryBackend after the
        # transfer; checks the tensor's backend struct directly rather
        # than relying on `backend_transfer/2` being a no-op for
        # same-backend transfers (an internal BinaryBackend behaviour).
        assert match?(%Nx.Tensor{data: %Nx.BinaryBackend{}}, actual)
        assert_close(actual, expected, tol: 1.0e-6)
      end
    end
  end

  describe "dequantize_defn/1 — composes inside defn" do
    import Nx.Defn

    defn dequantize_then_scale(qw, factor) do
      Emily.Quantization.dequantize_defn(qw) * factor
    end

    for bits <- @bits do
      test "bits=#{bits} runs under Nx.Defn.jit" do
        bits = unquote(bits)

        w =
          Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(128.0)
          |> Nx.subtract(0.5)

        qw = QuantizedWeight.from_dense(w, group_size: 64, bits: bits)
        factor = Nx.tensor(2.0, backend: Emily.Backend, type: :f32)

        actual = dequantize_then_scale(qw, factor)
        expected = QuantizedWeight.to_dense(qw) |> Nx.multiply(2.0)

        assert Nx.shape(actual) == {2, 128}
        assert_close(actual, expected, tol: 1.0e-6)
      end
    end

    test "mxfp4 runs under Nx.Defn.jit" do
      w =
        Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(128.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 32, bits: 4)
      # mxfp4 output is bf16; use a bf16 scalar so the multiply doesn't
      # promote and diverge from to_dense's bf16 path.
      factor = Nx.tensor(2.0, backend: Emily.Backend, type: {:bf, 16})

      actual = dequantize_then_scale(qw, factor)
      expected = QuantizedWeight.to_dense(qw) |> Nx.multiply(2.0)

      assert Nx.shape(actual) == {2, 128}
      assert_close(actual, expected, tol: 0.0)
    end
  end

  describe "dequantize_defn/1 — validation" do
    test "raises on mxfp8 / nvfp4 modes (still defn-unsupported)" do
      # mxfp4 is now defn-supported; the other two microscaled modes
      # remain on the Native path until a follow-up wires their LUTs.
      for {mode, group_size, bits} <- [{"mxfp8", 32, 8}, {"nvfp4", 16, 4}] do
        w = Nx.iota({2, 128}, backend: Emily.Backend, type: :f32) |> Nx.divide(256.0)
        qw = QuantizedWeight.from_dense(w, mode: mode, group_size: group_size, bits: bits)

        assert_raise ArgumentError, ~r/mode=.*#{mode}.*to_dense/, fn ->
          Quantization.dequantize_defn(qw)
        end
      end
    end

    test "raises on bits outside @defn_supported_bits via hand-built struct" do
      # `QuantizedWeight.from_dense/2` rejects bits ∉ [2,3,4,6,8] before
      # the defn validator can fire, so the only reachable trigger for
      # `validate_defn_bits!/1` is a hand-constructed struct. Cover the
      # raise path so the validator stays load-bearing.
      qw = %QuantizedWeight{
        value: Nx.tensor([[0]], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([[1.0]], type: :f32, backend: Emily.Backend),
        biases: Nx.tensor([[0.0]], type: :f32, backend: Emily.Backend),
        group_size: 1,
        bits: 5,
        transpose: true,
        mode: "affine"
      }

      assert_raise ArgumentError, ~r/bits=5 is not supported by the defn-native path/, fn ->
        Quantization.dequantize_defn(qw)
      end
    end
  end

  describe "dequantize_defn/1 — mxfp4" do
    test "matches QuantizedWeight.to_dense/1 on rank-2 input" do
      w =
        Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(64.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 32, bits: 4)

      actual = Quantization.dequantize_defn(qw)
      expected = QuantizedWeight.to_dense(qw)

      assert Nx.shape(actual) == Nx.shape(expected)
      assert Nx.type(actual) == {:bf, 16}
      assert Nx.type(expected) == {:bf, 16}
      # FP4 lane LUT entries and E8M0 scale powers are all exact in
      # bf16, so the defn path is bit-identical to MLX's NIF dequant
      # on every realistic scale value.
      assert_close(actual, expected, tol: 0.0)
    end

    test "matches QuantizedWeight.to_dense/1 on rank-3 input" do
      w =
        Nx.iota({2, 3, 128}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(128.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 32, bits: 4)

      actual = Quantization.dequantize_defn(qw)
      expected = QuantizedWeight.to_dense(qw)

      assert Nx.shape(actual) == {2, 3, 128}
      assert_close(actual, expected, tol: 0.0)
    end

    test "works after transfer to Nx.BinaryBackend" do
      w =
        Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
        |> Nx.divide(64.0)
        |> Nx.subtract(0.5)

      qw = QuantizedWeight.from_dense(w, mode: "mxfp4", group_size: 32, bits: 4)
      expected = QuantizedWeight.to_dense(qw) |> Nx.backend_transfer(Nx.BinaryBackend)

      actual =
        qw
        |> Nx.backend_transfer(Nx.BinaryBackend)
        |> Quantization.dequantize_defn()

      assert Nx.shape(actual) == Nx.shape(expected)
      assert match?(%Nx.Tensor{data: %Nx.BinaryBackend{}}, actual)
      assert_close(actual, expected, tol: 0.0)
    end
  end

  describe "dequantize_defn/1 — mxfp4 hand-packed oracle" do
    # Independent oracle: hand-pack a known FP4 lane sequence + a known
    # E8M0 scale byte and compare against pure-Elixir LUT decoding —
    # no MLX NIF in the comparison path. Breaks the shared-buffer
    # circularity of the round-trip-vs-to_dense tests.

    @fp4_lut [
      +0.0,
      +0.5,
      +1.0,
      +1.5,
      +2.0,
      +3.0,
      +4.0,
      +6.0,
      -0.0,
      -0.5,
      -1.0,
      -1.5,
      -2.0,
      -3.0,
      -4.0,
      -6.0
    ]

    test "mxfp4: dequantize_defn matches pure-Elixir LUT decode" do
      # 4 u32 = 32 lanes (lpu=8) = one group of 32. Lane codes
      # [0,1,2,...,15,0,1,2,...,15] sweep every FP4 code. Scale byte
      # 0x7F = 127 → 2^0 = 1.0, so dequant returns the raw FP4_LUT
      # values pointwise.
      packed = [0x76543210, 0xFEDCBA98, 0x76543210, 0xFEDCBA98]
      scale_byte = 0x7F

      expected =
        Enum.flat_map(0..1, fn _ ->
          Enum.map(0..15, &Enum.at(@fp4_lut, &1))
        end)

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([[scale_byte]], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 32,
        bits: 4,
        transpose: true,
        mode: "mxfp4"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end

    test "mxfp4: per-group scales apply independently" do
      # Two groups: 8 packed u32 = 64 lanes = 2 groups of 32 lanes
      # each (group_size is pinned at 32 for mxfp4). Distinct scales
      # 0x7F → 1.0 and 0x80 → 2.0 across the two groups, with
      # identical lane codes in each group so any output divergence
      # isolates to the scale multiplication.
      packed_one_group = [0x76543210, 0xFEDCBA98, 0x76543210, 0xFEDCBA98]
      packed = packed_one_group ++ packed_one_group
      scales_bytes = [0x7F, 0x80]
      scales_decoded = [1.0, 2.0]

      lanes_per_group =
        Enum.flat_map(0..1, fn _ -> Enum.map(0..15, &Enum.at(@fp4_lut, &1)) end)

      expected =
        Enum.flat_map(scales_decoded, fn s ->
          Enum.map(lanes_per_group, &(&1 * s))
        end)

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([scales_bytes], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 32,
        bits: 4,
        transpose: true,
        mode: "mxfp4"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end
  end

  describe "dequantize_defn/1 — hand-packed cross-word oracle" do
    # Independent oracle for the bits ∈ {3, 6} unpack path. Constructs
    # a packed u32 buffer with hand-chosen values, then dequantizes via
    # both `dequantize_defn` and a pure-Elixir Bitwise extractor. Breaks
    # the shared-buffer circularity of the round-trip-vs-to_dense tests
    # (both of which consume the same MLX-produced packed buffer with
    # the same lane convention — a coordinated lane-order bug would
    # pass both).

    import Bitwise

    test "bits=3: dequantize_defn matches pure-Bitwise lane extraction" do
      # 3 u32 = 96 bits = 32 lanes for bits=3. orig_last = 32; with
      # group_size = 16 we get 2 groups of 16 lanes each. Scales = 1.0
      # and biases = 0.0 so dequant just returns the lane integers as
      # floats, which we can compare directly.
      packed = [0xDEADBEEF, 0xCAFEBABE, 0x12345678]
      bits = 3
      group_size = 16

      # Pure-Elixir oracle: pack u32s into a single 96-bit integer
      # (little-endian within and across words), then slice 3 bits at a
      # time. This is the documented MLX packing convention.
      bitstream =
        packed
        |> Enum.with_index()
        |> Enum.reduce(0, fn {word, i}, acc -> acc ||| word <<< (32 * i) end)

      expected_lanes = for i <- 0..31, do: bitstream >>> (i * bits) &&& 0x7
      expected = expected_lanes |> Enum.map(&(&1 * 1.0)) |> List.wrap()

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.broadcast(1.0, {1, 2}) |> Nx.backend_transfer(Emily.Backend),
        biases: Nx.broadcast(0.0, {1, 2}) |> Nx.backend_transfer(Emily.Backend),
        group_size: group_size,
        bits: bits,
        transpose: true,
        mode: "affine"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end

    test "bits=6: dequantize_defn matches pure-Bitwise lane extraction" do
      # 3 u32 = 96 bits = 16 lanes for bits=6. orig_last = 16; group_size
      # = 16 gives one group. Pure-Elixir oracle as in the bits=3 case.
      packed = [0x0BADF00D, 0xFEEDFACE, 0x8BADF00D]
      bits = 6
      group_size = 16

      bitstream =
        packed
        |> Enum.with_index()
        |> Enum.reduce(0, fn {word, i}, acc -> acc ||| word <<< (32 * i) end)

      expected_lanes = for i <- 0..15, do: bitstream >>> (i * bits) &&& 0x3F
      expected = Enum.map(expected_lanes, &(&1 * 1.0))

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.broadcast(1.0, {1, 1}) |> Nx.backend_transfer(Emily.Backend),
        biases: Nx.broadcast(0.0, {1, 1}) |> Nx.backend_transfer(Emily.Backend),
        group_size: group_size,
        bits: bits,
        transpose: true,
        mode: "affine"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end
  end
end
