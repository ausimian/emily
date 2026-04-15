defmodule Emily.Soak.TrainingTest do
  @moduledoc """
  Memory-stability soak for the training-step hot path (M9 Phase E).
  Tagged `:soak`, stays in the default suite (finishes in ~1 s), same
  convention as `memory_test.exs`.

  Training exercises a different allocator pattern than inference:
  each step allocates a param tensor + its gradient + intermediate
  activations, then releases the old param tensor when the new one
  replaces it in the params map. Any retained grad ref, activation
  ref, or stale param ref would accumulate as a linear leak.

  Design mirrors `memory_test.exs` — warmup to amortize MLX's
  allocator setup, sample baseline after warmup, run N steps, assert
  memory returns within tolerance. Reuses the handwritten MLP from
  `TrainingHelper` (same module Phase D uses) so a regression here
  also points at the same params/grads the curve-matching test
  stresses numerically.
  """

  use ExUnit.Case, async: false

  alias Emily.Native
  alias Emily.TrainingHelper, as: TH

  @moduletag :soak

  @dims {4, 8, 3}
  @batch_shape {16, 4, 3}
  @iters 1_000
  @warmup 20

  # Tight vs `memory_test.exs` because each training step allocates
  # KB-scale tensors, not MB. A full retained param set + grads per
  # iteration over 1k steps would exceed ~2 MB by an order of
  # magnitude; 2 MB absorbs Metal arena drift without masking a leak
  # of more than a handful of retained refs.
  @tolerance_bytes 2 * 1024 * 1024

  defp workload(params, args) do
    {new_params, _loss} =
      Nx.Defn.jit_apply(
        &TH.mlp_step_with_loss/4,
        [params | args],
        compiler: Emily.Compiler
      )

    new_params
  end

  test "1k MLP training steps return to baseline" do
    params = TH.init_mlp(@dims, 0, Emily.Backend)
    {x, y} = TH.mlp_batch(@batch_shape, Emily.Backend)
    lr = Nx.tensor(0.5, type: {:f, 32}, backend: Emily.Backend)
    args = [x, y, lr]

    # Warmup — first JIT compile + allocator setup.
    Enum.reduce(1..@warmup, params, fn _i, p -> workload(p, args) end)

    :erlang.garbage_collect()
    Native.clear_cache()
    baseline = Native.get_active_memory()
    Native.reset_peak_memory()

    # Discard returned params each step — we're measuring the
    # per-iteration leak rate of the allocator, not accuracy.
    Enum.reduce(1..@iters, params, fn _i, p -> workload(p, args) end)

    :erlang.garbage_collect()
    Native.clear_cache()

    final = Native.get_active_memory()
    peak = Native.get_peak_memory()
    delta = final - baseline

    assert delta <= @tolerance_bytes,
           """
           training soak active-memory delta #{delta} bytes exceeds #{@tolerance_bytes}
             baseline:       #{baseline}
             final:          #{final}
             peak:           #{peak}
             iters:          #{@iters}
           A per-iter leak of one param set (~#{param_bytes()} bytes) over
           #{@iters} iters would add ~#{@iters * param_bytes()} bytes
           (#{div(@iters * param_bytes(), 1024)} KB).
           """
  end

  defp param_bytes do
    {in_dim, hidden, out_dim} = @dims
    (in_dim * hidden + hidden + hidden * out_dim + out_dim) * 4
  end
end
