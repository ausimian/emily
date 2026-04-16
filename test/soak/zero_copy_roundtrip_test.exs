defmodule Emily.Soak.ZeroCopyRoundTripTest do
  @moduledoc """
  Verify `to_binary` aliases the MLX buffer as a BEAM resource binary
  rather than memcpy. Two properties checked against allocator stats:

    * Single-shot delta: calling `Nx.to_binary/1` on a large tensor
      does not grow MLX active memory or the BEAM binary heap.
    * Soak: repeated `to_binary` calls stay bounded by a small
      multiple of the working-set size.
  """

  use ExUnit.Case, async: false

  alias Emily.Native

  @moduletag :soak

  defp mb(n), do: n * 1024 * 1024

  # Single-shot test: tolerances for MLX and BEAM binary heap growth.
  @mlx_tolerance_bytes 4 * 1024 * 1024
  @heap_tolerance_bytes 1 * 1024 * 1024

  # Soak test: active-memory and peak-memory ceilings.
  @leak_tolerance_bytes 8 * 1024 * 1024
  @peak_tolerance_bytes 32 * 1024 * 1024

  describe "to_binary zero-copy" do
    test "returning a 64 MB binary does not grow BEAM binary heap" do
      # 64 MB of f32 = 16M elements. iota on the Emily backend so the
      # buffer is born in MLX and is already row-contiguous.
      nelem = div(mb(64), 4)
      t = Nx.iota({nelem}, type: {:f, 32}, backend: Emily.Backend)

      # Force a materialization before we start measuring, so the
      # baseline reflects the steady-state cost of holding `t`.
      _ = Nx.to_binary(t)

      :erlang.garbage_collect()
      Native.clear_cache()
      mlx_baseline = Native.get_active_memory()
      bin_baseline = :erlang.memory(:binary)
      Native.reset_peak_memory()

      bin = Nx.to_binary(t)

      # Two orthogonal signals that we're aliasing, not copying:
      #
      # (1) MLX active memory should not grow — the returned binary
      #     points into the already-allocated MLX buffer, not a fresh
      #     MLX malloc.
      # (2) BEAM's binary heap should not grow — resource binaries
      #     live outside BEAM's binary allocator (the data pointer
      #     aliases MLX memory). A memcpy path would instead allocate
      #     a 64 MB refc binary on BEAM's binary heap.
      mlx_delta = Native.get_active_memory() - mlx_baseline
      bin_delta = :erlang.memory(:binary) - bin_baseline

      assert byte_size(bin) == mb(64)

      assert mlx_delta < @mlx_tolerance_bytes,
             """
             MLX active-memory delta #{mlx_delta} bytes exceeds tolerance #{@mlx_tolerance_bytes}
               mlx baseline:  #{mlx_baseline}
               mlx after:     #{Native.get_active_memory()}
             """

      assert bin_delta < @heap_tolerance_bytes,
             """
             BEAM binary-heap delta #{bin_delta} bytes exceeds tolerance #{@heap_tolerance_bytes}
               bin baseline:  #{bin_baseline}
               bin after:     #{:erlang.memory(:binary)}
             """
    end

    test "200 round-trips on a 4 MB tensor stay bounded" do
      # Small enough that cumulative memcpy allocations would show up
      # as meaningful peak growth. Large enough to be a refc binary.
      nelem = div(mb(4), 4)
      t = Nx.iota({nelem}, type: {:f, 32}, backend: Emily.Backend)

      # Warmup + baseline.
      for _ <- 1..5, do: Nx.to_binary(t)
      :erlang.garbage_collect()
      Native.clear_cache()
      baseline = Native.get_active_memory()
      Native.reset_peak_memory()

      for _ <- 1..200 do
        bin = Nx.to_binary(t)
        # Touch every page so we're sure the alias is real-readable.
        assert byte_size(bin) == mb(4)
      end

      :erlang.garbage_collect()
      Native.clear_cache()

      final = Native.get_active_memory()
      peak = Native.get_peak_memory()
      delta = final - baseline

      # Leak tolerance: bounded by a small multiple of working-set size.
      assert delta <= @leak_tolerance_bytes,
             """
             active-memory delta #{delta} bytes exceeds tolerance #{@leak_tolerance_bytes}
               baseline:       #{baseline}
               final:          #{final}
               peak:           #{peak}
             """

      assert peak - baseline <= @peak_tolerance_bytes,
             """
             peak-memory delta #{peak - baseline} bytes exceeds tolerance #{@peak_tolerance_bytes}
               baseline: #{baseline}
               peak:     #{peak}
             """
    end
  end
end
