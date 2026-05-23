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

    for {mode, group_size, bits} <- [{"mxfp4", 32, 4}, {"mxfp8", 32, 8}, {"nvfp4", 16, 4}] do
      test "#{mode} runs under Nx.Defn.jit" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        w =
          Nx.iota({2, 128}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(128.0)
          |> Nx.subtract(0.5)

        qw = QuantizedWeight.from_dense(w, mode: mode, group_size: group_size, bits: bits)
        # Microscaled output is bf16; use a bf16 scalar so the multiply
        # doesn't promote and diverge from to_dense's bf16 path.
        factor = Nx.tensor(2.0, backend: Emily.Backend, type: {:bf, 16})

        actual = dequantize_then_scale(qw, factor)
        expected = QuantizedWeight.to_dense(qw) |> Nx.multiply(2.0)

        assert Nx.shape(actual) == {2, 128}
        assert_close(actual, expected, tol: 0.0)
      end
    end
  end

  describe "dequantize_defn/1 — validation" do
    test "raises on unknown mode via hand-built struct" do
      # All MLX-supported modes (affine, mxfp4, mxfp8, nvfp4) are now
      # defn-wired, so `QuantizedWeight.from_dense/2` cannot produce an
      # unsupported-mode struct. Hand-construct one so the catch-all
      # `validate_defn_mode!/1` clause stays load-bearing against a
      # future MLX mode addition or a caller passing a typo.
      qw = %QuantizedWeight{
        value: Nx.tensor([[0]], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([[0]], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 1,
        bits: 4,
        transpose: true,
        mode: "fakemode"
      }

      assert_raise ArgumentError, ~r/mode=.*fakemode.*to_dense/, fn ->
        Quantization.dequantize_defn(qw)
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

  describe "dequantize_defn/1 — microscaled (mxfp4, mxfp8, nvfp4)" do
    # The three modes share lane-decode infrastructure: mxfp4 and
    # nvfp4 use the 16-entry FP4-E2M1 lane LUT, mxfp8 uses the
    # 256-entry FP8-E4M3 lane LUT. Scale decode splits the other way:
    # mxfp4 and mxfp8 share the 256-entry FP8-E8M0 scale LUT (one
    # exponent byte per group), while nvfp4 reuses the FP8-E4M3 LUT
    # for its per-group scale bytes. Every LUT entry is exact in
    # bf16, so the defn path is bit-identical to MLX's NIF dequant
    # on realistic scale values.
    for {mode, group_size, bits} <- [{"mxfp4", 32, 4}, {"mxfp8", 32, 8}, {"nvfp4", 16, 4}] do
      test "#{mode}: matches QuantizedWeight.to_dense/1 on rank-2 input" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        w =
          Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(64.0)
          |> Nx.subtract(0.5)

        qw = QuantizedWeight.from_dense(w, mode: mode, group_size: group_size, bits: bits)

        actual = Quantization.dequantize_defn(qw)
        expected = QuantizedWeight.to_dense(qw)

        assert Nx.shape(actual) == Nx.shape(expected)
        assert Nx.type(actual) == {:bf, 16}
        assert Nx.type(expected) == {:bf, 16}
        assert_close(actual, expected, tol: 0.0)
      end

      test "#{mode}: matches QuantizedWeight.to_dense/1 on rank-3 input" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        w =
          Nx.iota({2, 3, 128}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(128.0)
          |> Nx.subtract(0.5)

        qw = QuantizedWeight.from_dense(w, mode: mode, group_size: group_size, bits: bits)

        actual = Quantization.dequantize_defn(qw)
        expected = QuantizedWeight.to_dense(qw)

        assert Nx.shape(actual) == {2, 3, 128}
        assert_close(actual, expected, tol: 0.0)
      end

      test "#{mode}: works after transfer to Nx.BinaryBackend" do
        mode = unquote(mode)
        group_size = unquote(group_size)
        bits = unquote(bits)

        w =
          Nx.iota({4, 64}, backend: Emily.Backend, type: :f32)
          |> Nx.divide(64.0)
          |> Nx.subtract(0.5)

        qw = QuantizedWeight.from_dense(w, mode: mode, group_size: group_size, bits: bits)
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

  describe "dequantize_defn/1 — mxfp8 hand-packed oracle" do
    # Independent oracle for the bits=8 FP8-E4M3 lane decode. Uses a
    # set of well-known E4M3 byte codes whose decoded values are
    # documented in the OCP MX format spec (and verified against MLX's
    # bit-trick decode) so the test doesn't depend on either the
    # production LUT or MLX's NIF.

    import Bitwise

    # FP8-E4M3 codes where sign=0, exp = k, mant=0 decode to 2^(k-7)
    # before the FromFP8 bit-trick's *256 multiplier. With the *256
    # baked in, value = 2^(k+1). So byte 0x38 (exp=7, mant=0) → 1.0,
    # 0x40 (exp=8) → 2.0, 0x48 (exp=9) → 4.0, 0x50 (exp=10) → 8.0.
    @e4m3_codes [0x38, 0x40, 0x48, 0x50]
    @e4m3_values [1.0, 2.0, 4.0, 8.0]

    test "mxfp8: dequantize_defn matches hand-computed values per FP8-E4M3 spec" do
      # Pack 32 lanes = 8 packed u32s, each holding the same 4 codes
      # in little-endian byte order. group_size=32 → one group.
      [b0, b1, b2, b3] = @e4m3_codes
      packed_word = b0 ||| b1 <<< 8 ||| b2 <<< 16 ||| b3 <<< 24
      packed = List.duplicate(packed_word, 8)
      scale_byte = 0x7F

      lane_values = Enum.flat_map(1..8, fn _ -> @e4m3_values end)
      expected = lane_values

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([[scale_byte]], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 32,
        bits: 8,
        transpose: true,
        mode: "mxfp8"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end

    test "mxfp8: per-group scales apply independently" do
      # Two groups (64 lanes, 16 packed u32). Same lane pattern in
      # each group; distinct scales 0x7F (= 1.0) and 0x80 (= 2.0)
      # broadcast per group. Negative lanes (codes 0xB8, 0xC0, 0xC8,
      # 0xD0 = sign-flipped 0x38, 0x40, 0x48, 0x50) verify the sign
      # bit threads through the FP8-E4M3 LUT.
      [b0, b1, b2, b3] = @e4m3_codes
      [n0, n1, n2, n3] = Enum.map(@e4m3_codes, &(&1 ||| 0x80))

      pos_word = b0 ||| b1 <<< 8 ||| b2 <<< 16 ||| b3 <<< 24
      neg_word = n0 ||| n1 <<< 8 ||| n2 <<< 16 ||| n3 <<< 24

      # Group 0: 32 lanes mixing positives and negatives.
      # Group 1: same pattern. Scales 1.0 and 2.0 distinguish them.
      one_group = List.duplicate(pos_word, 4) ++ List.duplicate(neg_word, 4)
      packed = one_group ++ one_group
      scales_bytes = [0x7F, 0x80]

      pos_lanes = Enum.flat_map(1..4, fn _ -> @e4m3_values end)
      neg_lanes = Enum.flat_map(1..4, fn _ -> Enum.map(@e4m3_values, &(-&1)) end)
      lanes_per_group = pos_lanes ++ neg_lanes

      expected =
        Enum.flat_map([1.0, 2.0], fn s ->
          Enum.map(lanes_per_group, &(&1 * s))
        end)

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([scales_bytes], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 32,
        bits: 8,
        transpose: true,
        mode: "mxfp8"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end
  end

  describe "dequantize_defn/1 — nvfp4 hand-packed oracle" do
    # nvfp4 reuses the FP4-E2M1 lane LUT (same as mxfp4) but consumes
    # FP8-E4M3 per-group scale bytes (same encoding as mxfp8 lanes).
    # Verifies that the production code correctly composes the two LUTs
    # — a coordinated bug in either decoder would still be caught here.

    @fp4_lut_nv [
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

    test "nvfp4: dequantize_defn matches hand-computed values per FP4 LUT" do
      # 2 packed u32 = 16 lanes = one group (group_size = 16). Scale
      # byte 0x38 = FP8-E4M3 unity (1.0), so dequant returns the raw
      # FP4 LUT entries pointwise.
      packed = [0x76543210, 0xFEDCBA98]
      scale_byte = 0x38

      expected = @fp4_lut_nv

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([[scale_byte]], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 16,
        bits: 4,
        transpose: true,
        mode: "nvfp4"
      }

      actual = qw |> Quantization.dequantize_defn() |> Nx.to_flat_list()
      assert actual == expected
    end

    test "nvfp4: FP8-E4M3 scales broadcast independently per group" do
      # Two groups (4 packed u32 = 32 lanes = 2 × group_size 16), with
      # FP8-E4M3 scale bytes 0x38 (= 1.0) and 0x40 (= 2.0). Lane codes
      # identical in both groups so any output divergence isolates to
      # the FP8-E4M3 scale decode + multiplication.
      packed_one_group = [0x76543210, 0xFEDCBA98]
      packed = packed_one_group ++ packed_one_group
      scales_bytes = [0x38, 0x40]
      scales_decoded = [1.0, 2.0]

      expected =
        Enum.flat_map(scales_decoded, fn s ->
          Enum.map(@fp4_lut_nv, &(&1 * s))
        end)

      qw = %QuantizedWeight{
        value: Nx.tensor([packed], type: :u32, backend: Emily.Backend),
        scales: Nx.tensor([scales_bytes], type: {:u, 8}, backend: Emily.Backend),
        biases: Nx.tensor(0.0, type: :f32, backend: Emily.Backend),
        group_size: 16,
        bits: 4,
        transpose: true,
        mode: "nvfp4"
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
