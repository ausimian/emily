defmodule EmilyTest do
  use ExUnit.Case, async: true
  doctest Emily

  describe "from_binary/3 + to_binary/1 — round-trip" do
    test "f32 scalar" do
      bin = <<1.5::float-32-native>>
      t = Emily.from_binary(bin, [1], {:f, 32})

      assert Emily.to_binary(t) == bin
      assert Emily.shape(t) == [1]
      assert Emily.dtype(t) == {:f, 32}
    end

    test "f32 1-D vector" do
      bin = for x <- 1..8, into: <<>>, do: <<x * 1.0::float-32-native>>
      t = Emily.from_binary(bin, [8], {:f, 32})

      assert Emily.to_binary(t) == bin
      assert Emily.shape(t) == [8]
    end

    test "f32 2-D matrix" do
      bin =
        for x <- 1..12, into: <<>>, do: <<x * 1.0::float-32-native>>

      t = Emily.from_binary(bin, [3, 4], {:f, 32})

      assert Emily.to_binary(t) == bin
      assert Emily.shape(t) == [3, 4]
      assert Emily.dtype(t) == {:f, 32}
    end

    test "s64 1-D vector" do
      bin = for x <- -3..4, into: <<>>, do: <<x::signed-integer-64-native>>
      t = Emily.from_binary(bin, [8], {:s, 64})

      assert Emily.to_binary(t) == bin
      assert Emily.dtype(t) == {:s, 64}
    end

    test "u8 1-D vector" do
      bin = <<1, 2, 3, 4, 5, 6, 7, 8>>
      t = Emily.from_binary(bin, [8], {:u, 8})

      assert Emily.to_binary(t) == bin
      assert Emily.dtype(t) == {:u, 8}
    end

    test "pred 1-D vector" do
      # MLX bool_ is stored as 1 byte per element.
      bin = <<1, 0, 1, 1, 0>>
      t = Emily.from_binary(bin, [5], {:pred, 1})

      assert Emily.to_binary(t) == bin
      assert Emily.dtype(t) == {:pred, 1}
    end
  end

  describe "error paths" do
    test "binary size mismatch raises" do
      # Claim a shape of [4] f32 (16 bytes) but supply only 4 bytes.
      bin = <<1.0::float-32-native>>

      assert_raise ArgumentError, ~r/binary size mismatch/, fn ->
        Emily.from_binary(bin, [4], {:f, 32})
      end
    end

    test "unsupported dtype raises" do
      # f64 is not supported by MLX (no Metal f64).
      bin = <<1.0::float-64-native>>

      assert_raise ArgumentError, ~r/unsupported dtype/, fn ->
        Emily.from_binary(bin, [1], {:f, 64})
      end
    end
  end

  describe "lifecycle" do
    test "eval/1 is idempotent and does not change bytes" do
      bin = for x <- 1..16, into: <<>>, do: <<x * 1.0::float-32-native>>
      t = Emily.from_binary(bin, [4, 4], {:f, 32})

      assert :ok = Emily.eval(t)
      assert :ok = Emily.eval(t)
      assert Emily.to_binary(t) == bin
    end

    test "tensor survives GC as long as a ref is held" do
      bin = <<42.0::float-32-native>>
      t = Emily.from_binary(bin, [1], {:f, 32})

      # Provoke a GC; the resource should still be valid.
      :erlang.garbage_collect()
      :erlang.garbage_collect(self())

      assert Emily.to_binary(t) == bin
    end

    test "to_binary aliased binary survives after tensor goes out of scope" do
      # to_binary returns a resource binary that aliases MLX storage.
      # If the refcount wiring is wrong, reading after the tensor is
      # dropped would segfault or return garbage.
      bin =
        for x <- 1..1024, into: <<>>, do: <<x * 1.0::float-32-native>>

      out =
        (fn ->
           t = Emily.from_binary(bin, [1024], {:f, 32})
           Emily.to_binary(t)
         end).()

      :erlang.garbage_collect()
      :erlang.garbage_collect(self())

      assert out == bin
    end
  end
end
