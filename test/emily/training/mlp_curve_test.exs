defmodule Emily.Training.MlpCurveTest do
  @moduledoc """
  Training-curve matching for a handwritten 2-layer MLP (M9 Phase D).

  Runs 50 SGD steps on a fixed synthetic dataset with deterministic
  initial weights, collects per-step losses under `Emily.Compiler`
  and `Nx.Defn.Evaluator` on `Nx.BinaryBackend`, and asserts two
  things:

    * **Per-step rtol = 1e-3** (silent-drift canary). MLX's parallel
      reductions produce a different rounding path than BinaryBackend's
      sequential reductions; strict bit-match would fail. 1e-3 is tight
      enough to detect a divergent trajectory from ~step 10 onward, but
      loose enough to absorb f32 reduction-order drift.
    * **Final-loss rtol = 1e-4** (convergence correctness). By step
      50 the loss has concentrated most of its magnitude away from
      initial noise; a tighter bar catches cases where per-step drift
      averaged out but the optimizer ended up somewhere wrong.

  No Axon dependency — handwritten MLP + SGD via `Nx.Defn.grad`.
  Axon shows up in Phase F's `:training_full` MNIST canary.
  """

  use ExUnit.Case, async: true

  alias Emily.TrainingHelper, as: TH

  @dims {4, 8, 3}
  @batch_shape {16, 4, 3}
  @steps 50
  @lr_val 0.5

  test "per-step loss trajectory matches BinaryBackend within tolerance" do
    # BinaryBackend side — the oracle.
    params_bin = TH.init_mlp(@dims, 0, Nx.BinaryBackend)
    {x_bin, y_bin} = TH.mlp_batch(@batch_shape, Nx.BinaryBackend)
    lr_bin = Nx.tensor(@lr_val, type: {:f, 32}, backend: Nx.BinaryBackend)

    losses_bin =
      TH.run_steps(
        &TH.mlp_step_with_loss/4,
        params_bin,
        [x_bin, y_bin, lr_bin],
        @steps,
        Nx.Defn.Evaluator
      )

    # Emily side — bit-identical initial params/data, but graph built
    # on Emily.Backend and compiled via Emily.Compiler.
    params_emily = TH.init_mlp(@dims, 0, Emily.Backend)
    {x_emily, y_emily} = TH.mlp_batch(@batch_shape, Emily.Backend)
    lr_emily = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)

    losses_emily =
      TH.run_steps(
        &TH.mlp_step_with_loss/4,
        params_emily,
        [x_emily, y_emily, lr_emily],
        @steps,
        Emily.Compiler
      )

    assert length(losses_emily) == @steps
    assert length(losses_bin) == @steps

    # Sanity: loss actually decreased over the run. The synthetic
    # task is intentionally flat (MSE between two independently-
    # sampled det_weights tensors) so we don't assert a magnitude
    # of decrease — the curve-match is the real test.
    assert List.first(losses_bin) > List.last(losses_bin)

    # Per-step match — silent-drift canary.
    for {{le, lb}, i} <- Enum.zip(losses_emily, losses_bin) |> Enum.with_index() do
      close?(le, lb, 1.0e-4, 1.0e-3) ||
        flunk_trajectory(i, le, lb, losses_emily, losses_bin)
    end

    # Final-loss match — convergence correctness.
    le_final = List.last(losses_emily)
    lb_final = List.last(losses_bin)

    assert close?(le_final, lb_final, 1.0e-5, 1.0e-4),
           "final loss divergence: emily=#{le_final} bin=#{lb_final} " <>
             "reldiff=#{abs(le_final - lb_final) / abs(lb_final)}"
  end

  defp close?(a, b, atol, rtol), do: abs(a - b) <= atol + rtol * abs(b)

  defp flunk_trajectory(i, le, lb, losses_emily, losses_bin) do
    preview_e = losses_emily |> Enum.take(min(i + 3, length(losses_emily)))
    preview_b = losses_bin |> Enum.take(min(i + 3, length(losses_bin)))

    flunk("""
    per-step loss diverged at step #{i}:
      emily=#{le} bin=#{lb} reldiff=#{abs(le - lb) / abs(lb)}
    emily trajectory (first #{length(preview_e)} steps): #{inspect(preview_e)}
    bin   trajectory (first #{length(preview_b)} steps): #{inspect(preview_b)}
    """)
  end
end
