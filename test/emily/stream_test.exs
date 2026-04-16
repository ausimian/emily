defmodule Emily.StreamTest do
  use ExUnit.Case, async: false

  describe "Emily.Stream" do
    test "new/1 creates a stream with a valid index" do
      stream = Emily.Stream.new(:gpu)
      assert is_integer(stream.index)
      assert stream.device == :gpu
    end

    test "new/1 creates distinct streams" do
      s1 = Emily.Stream.new(:gpu)
      s2 = Emily.Stream.new(:gpu)
      assert s1.index != s2.index
    end

    test "with_stream/2 scopes stream to block" do
      stream = Emily.Stream.new(:gpu)

      result =
        Emily.Stream.with_stream(stream, fn ->
          t = Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend)
          Nx.sum(t) |> Nx.to_number()
        end)

      assert result == 6.0
    end

    test "with_stream/2 stores and restores process dictionary" do
      assert Process.get(:emily_stream) == nil

      stream = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(stream, fn ->
        assert Process.get(:emily_stream) == stream.index
      end)

      assert Process.get(:emily_stream) == nil
    end

    test "nested with_stream restores correctly" do
      s1 = Emily.Stream.new(:gpu)
      s2 = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(s1, fn ->
        assert Process.get(:emily_stream) == s1.index

        Emily.Stream.with_stream(s2, fn ->
          assert Process.get(:emily_stream) == s2.index
        end)

        assert Process.get(:emily_stream) == s1.index
      end)

      assert Process.get(:emily_stream) == nil
    end

    test "with_stream/2 restores on exception" do
      stream = Emily.Stream.new(:gpu)

      assert_raise RuntimeError, fn ->
        Emily.Stream.with_stream(stream, fn ->
          raise "boom"
        end)
      end

      assert Process.get(:emily_stream) == nil
    end

    test "synchronize/1 completes without error" do
      stream = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(stream, fn ->
        _t = Nx.tensor([1.0, 2.0], backend: Emily.Backend) |> Nx.multiply(2.0)
      end)

      assert :ok == Emily.Stream.synchronize(stream)
    end

    test "computation on a stream produces correct results" do
      stream = Emily.Stream.new(:gpu)

      result =
        Emily.Stream.with_stream(stream, fn ->
          a = Nx.iota({4, 4}, type: {:f, 32}, backend: Emily.Backend)
          b = Nx.iota({4, 4}, type: {:f, 32}, backend: Emily.Backend)
          Nx.add(a, b) |> Nx.sum() |> Nx.to_number()
        end)

      # sum of 2 * iota({4,4}) = 2 * (0+1+...+15) = 2 * 120 = 240
      assert result == 240.0
    end
  end
end
