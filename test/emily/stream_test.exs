defmodule Emily.StreamTest do
  use ExUnit.Case, async: false

  describe "Emily.Stream" do
    test "new/1 creates a stream with a worker reference" do
      stream = Emily.Stream.new(:gpu)
      assert is_reference(stream.worker)
    end

    test "new/1 creates distinct streams" do
      s1 = Emily.Stream.new(:gpu)
      s2 = Emily.Stream.new(:gpu)
      assert s1.worker != s2.worker
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
      assert Process.get(:emily_worker) == nil

      stream = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(stream, fn ->
        assert Process.get(:emily_worker) == stream.worker
      end)

      assert Process.get(:emily_worker) == nil
    end

    test "nested with_stream restores correctly" do
      s1 = Emily.Stream.new(:gpu)
      s2 = Emily.Stream.new(:gpu)

      Emily.Stream.with_stream(s1, fn ->
        assert Process.get(:emily_worker) == s1.worker

        Emily.Stream.with_stream(s2, fn ->
          assert Process.get(:emily_worker) == s2.worker
        end)

        assert Process.get(:emily_worker) == s1.worker
      end)

      assert Process.get(:emily_worker) == nil
    end

    test "with_stream/2 restores on exception" do
      stream = Emily.Stream.new(:gpu)

      assert_raise RuntimeError, fn ->
        Emily.Stream.with_stream(stream, fn ->
          raise "boom"
        end)
      end

      assert Process.get(:emily_worker) == nil
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
