defmodule Emily.Backend.LifecycleTest do
  @moduledoc """
  Tests for backend ownership: from_binary/to_binary round-trip,
  backend_copy/transfer to and from BinaryBackend, inspect, to_batched.
  """

  use ExUnit.Case, async: true

  describe "init/1" do
    test "accepts empty options" do
      assert Emily.Backend.init([]) == [device: :gpu]
    end

    test "accepts :cpu and :gpu devices" do
      assert Emily.Backend.init(device: :cpu) == [device: :cpu]
      assert Emily.Backend.init(device: :gpu) == [device: :gpu]
    end

    test "rejects unknown options" do
      assert_raise ArgumentError, fn ->
        Emily.Backend.init(bogus: true)
      end
    end

    test "rejects unsupported device" do
      assert_raise ArgumentError, ~r/:device to be :cpu or :gpu/, fn ->
        Emily.Backend.init(device: :tpu)
      end
    end
  end

  describe "from_binary/3 and to_binary/2" do
    test "round-trips f32" do
      bin = <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>
      t = Nx.from_binary(bin, {:f, 32}, backend: Emily.Backend)
      assert Nx.to_binary(t) == bin
      assert Nx.shape(t) == {3}
      assert Nx.type(t) == {:f, 32}
    end

    test "round-trips s32" do
      bin = <<1::32-signed-native, 2::32-signed-native, -3::32-signed-native>>
      t = Nx.from_binary(bin, {:s, 32}, backend: Emily.Backend)
      assert Nx.to_binary(t) == bin
    end

    test "round-trips u8" do
      bin = <<1, 2, 3, 4>>
      t = Nx.from_binary(bin, {:u, 8}, backend: Emily.Backend)
      assert Nx.to_binary(t) == bin
    end

    test "honours limit on to_binary" do
      bin = <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>
      t = Nx.from_binary(bin, {:f, 32}, backend: Emily.Backend)
      assert byte_size(Nx.to_binary(t, limit: 2)) == 8
    end

    test "rejects f64" do
      assert_raise ArgumentError, ~r/does not support \{:f, 64\}/, fn ->
        bin = <<1.0::float-64-native>>
        Nx.from_binary(bin, {:f, 64}, backend: Emily.Backend)
      end
    end
  end

  describe "backend_copy / backend_transfer" do
    setup do
      %{tensor: Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Emily.Backend)}
    end

    test "backend_copy to self is identity-like", %{tensor: t} do
      copied = Nx.backend_copy(t, Emily.Backend)
      assert Nx.to_flat_list(copied) == [1.0, 2.0, 3.0, 4.0]
    end

    test "backend_transfer to BinaryBackend", %{tensor: t} do
      b = Nx.backend_transfer(t, Nx.BinaryBackend)
      assert b.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(b) == [1.0, 2.0, 3.0, 4.0]
    end

    test "backend_transfer from BinaryBackend", %{tensor: _} do
      b = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Nx.BinaryBackend)
      e = Nx.backend_transfer(b, Emily.Backend)
      assert e.data.__struct__ == Emily.Backend
      assert Nx.to_flat_list(e) == [1.0, 2.0, 3.0, 4.0]
    end

    test "backend_deallocate returns :ok", %{tensor: t} do
      assert Nx.backend_deallocate(t) == :ok
    end
  end

  describe "inspect/2" do
    test "renders tensor like other backends" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend)
      out = inspect(t)
      assert out =~ "f32[3]"
      assert out =~ "1.0"
      assert out =~ "3.0"
    end
  end

  describe "to_batched/3" do
    test "splits along axis 0 evenly" do
      t = Nx.iota({6, 2}, backend: Emily.Backend, type: {:f, 32})
      batches = Nx.to_batched(t, 2) |> Enum.to_list()
      assert length(batches) == 3

      assert Enum.map(batches, &Nx.to_flat_list/1) == [
               [0.0, 1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0, 7.0],
               [8.0, 9.0, 10.0, 11.0]
             ]
    end

    test "discards leftover when asked" do
      t = Nx.iota({5, 2}, backend: Emily.Backend, type: {:f, 32})
      batches = Nx.to_batched(t, 2, leftover: :discard) |> Enum.to_list()
      assert length(batches) == 2
    end

    test "repeats leftover by default" do
      t = Nx.iota({5, 2}, backend: Emily.Backend, type: {:f, 32})
      batches = Nx.to_batched(t, 2) |> Enum.to_list()
      assert length(batches) == 3
      # Last batch wraps: row 4 then row 0.
      assert Nx.to_flat_list(List.last(batches)) == [8.0, 9.0, 0.0, 1.0]
    end
  end

  describe "pointer interop" do
    test "to_pointer raises" do
      t = Nx.tensor([1.0], backend: Emily.Backend)

      assert_raise ArgumentError, ~r/does not implement pointer/, fn ->
        Emily.Backend.to_pointer(t, [])
      end
    end

    test "from_pointer raises" do
      assert_raise ArgumentError, ~r/does not implement pointer/, fn ->
        Emily.Backend.from_pointer(0, {:f, 32}, {1}, [], [])
      end
    end
  end

  # MLX has no instruction for either. Nx's population_count /
  # count_leading_zeros apply to small int tensors, and a workaround
  # would be pure CPU — not worth carrying until a caller needs it.
  describe "unsupported bitwise ops" do
    test "count_leading_zeros raises" do
      t = Nx.tensor([1, 2, 3], type: {:s, 32}, backend: Emily.Backend)

      assert_raise ArgumentError, ~r/count_leading_zeros/, fn ->
        Nx.count_leading_zeros(t)
      end
    end

    test "population_count raises" do
      t = Nx.tensor([1, 2, 3], type: {:s, 32}, backend: Emily.Backend)

      assert_raise ArgumentError, ~r/population_count/, fn ->
        Nx.population_count(t)
      end
    end
  end

  describe "pad" do
    # MLX pad supports low/high padding but no interior dilation. Reject
    # cleanly rather than silently dropping the interior spec.
    test "raises on interior padding" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Emily.Backend)
      pad_value = Nx.tensor(0.0, backend: Emily.Backend)

      assert_raise ArgumentError, ~r/interior padding/, fn ->
        Emily.Backend.pad(t, t, pad_value, [{0, 0, 1}, {0, 0, 0}])
      end
    end
  end

  describe "backend_transfer edge cases" do
    # Transferring to the meta-module `Nx.Tensor` is a no-op —
    # Nx.BinaryBackend treats it the same way. This keeps code that
    # calls `backend_transfer(t, Nx.Tensor)` portable across backends.
    test "to Nx.Tensor is identity" do
      t = Nx.tensor([1.0, 2.0], backend: Emily.Backend)
      same = Emily.Backend.backend_transfer(t, Nx.Tensor, [])
      assert same.data.__struct__ == Emily.Backend
    end
  end

  describe "from_binary iodata fallback" do
    # `Nx.from_binary`'s public API guards for `is_binary(binary)`, but
    # the backend callback also handles iodata. Call the backend
    # directly to exercise the iolist branch of `ensure_binary/1`.
    test "flattens an iolist of binaries" do
      bin1 = <<1.0::float-32-native, 2.0::float-32-native>>
      bin2 = <<3.0::float-32-native, 4.0::float-32-native>>

      out_template = %Nx.Tensor{shape: {4}, type: {:f, 32}, names: [nil], data: nil}
      t = Emily.Backend.from_binary(out_template, [bin1, bin2], [])
      assert Nx.to_flat_list(t) == [1.0, 2.0, 3.0, 4.0]
    end
  end

  describe "inspect/2 limit" do
    # Nx passes `limit: :infinity` when the caller wants the full
    # tensor. Exercises the :infinity branch of Emily.Backend.inspect.
    test "renders full tensor under :infinity" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend)
      doc = Emily.Backend.inspect(t, %Inspect.Opts{limit: :infinity, custom_options: []})
      rendered = doc |> Inspect.Algebra.format(80) |> IO.iodata_to_binary()
      assert rendered =~ "1.0"
      assert rendered =~ "3.0"
    end
  end

  describe "bitcast" do
    # MLX exposes `mx::view(array, dtype)` — a zero-copy reinterpret
    # cast between equal-width dtypes. Nx.Random uses this to move
    # between float and uint of matching bit width (e.g. f32 ↔ u32).
    test "reinterprets f32 bits as u32" do
      t = Nx.tensor([1.0, 2.0, -1.0], type: {:f, 32}, backend: Emily.Backend)

      emily = Nx.bitcast(t, {:u, 32})
      ref = Nx.tensor([1.0, 2.0, -1.0], type: {:f, 32}) |> Nx.bitcast({:u, 32})

      assert Nx.type(emily) == {:u, 32}
      assert Nx.to_flat_list(emily) == Nx.to_flat_list(ref)
    end

    test "reinterprets u32 bits as f32" do
      u = Nx.tensor([1_065_353_216, 1_073_741_824], type: {:u, 32}, backend: Emily.Backend)

      emily = Nx.bitcast(u, {:f, 32})

      assert Nx.type(emily) == {:f, 32}
      assert Nx.to_flat_list(emily) == [1.0, 2.0]
    end
  end
end
