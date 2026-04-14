defmodule Emily.Soak.MemoryTest do
  @moduledoc """
  Memory-stability soak test. Tagged `:soak` so it's excluded from the
  default `mix test` run. Execute with `mix test --only soak`.

  Allocates and drops many tensors in a tight loop, evaluating each,
  then asserts MLX's active-memory figure returns to the baseline once
  the BEAM garbage-collects our Tensor refs and we call
  `clear_cache/0`. A leak here would show up as a growing delta over
  iterations.
  """

  use ExUnit.Case, async: false

  import Emily.TensorHelpers

  alias Emily.Native

  @moduletag :soak

  @iters 5_000
  # Allow MLX allocator book-keeping to fluctuate by this many bytes
  # before we call it a leak. Empirically an M-series MLX allocator
  # sits at tens of KB of small persistent buffers.
  @tolerance_bytes 1_024 * 1_024

  test "repeated allocate/evaluate/drop returns to baseline" do
    # Drain any per-test scaffolding before taking the baseline.
    :erlang.garbage_collect()
    Native.clear_cache()
    baseline = Native.get_active_memory()

    Native.reset_peak_memory()

    for _ <- 1..@iters do
      a = f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [8])
      b = Native.multiply(a, a)
      _ = Native.to_binary(b)
      :ok
    end

    :erlang.garbage_collect()
    Native.clear_cache()

    final = Native.get_active_memory()
    peak = Native.get_peak_memory()
    delta = final - baseline

    assert delta <= @tolerance_bytes,
           """
           active-memory delta #{delta} bytes exceeds tolerance #{@tolerance_bytes}
             baseline: #{baseline}
             final:    #{final}
             peak:     #{peak}
             iters:    #{@iters}
           """
  end
end
