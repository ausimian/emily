defmodule Emily.Training.CnnCurveTest do
  @moduledoc """
  Training-curve matching for a small handwritten CNN (M17 Phase D).

  Model: conv(1→4, 3x3) → ReLU → maxpool(2x2) → conv(4→8, 3x3) →
  ReLU → maxpool(2x2) → flatten → dense(→3). SGD with MSE loss on a
  synthetic deterministic batch.

  Asserts per-step loss trajectory matches `Nx.BinaryBackend` within
  a dtype-aware tolerance. This is the M17 equivalent of
  `mlp_curve_test.exs`: it's the "real" regression gate for the
  native window ops end-to-end, because the grad of `window_max`
  lands on `window_scatter_max` during every backward pass.

  No Axon — the handwritten path keeps the failure surface tiny so a
  red curve here points at backend numerics (or the C++ pooling
  path), not at indirect Axon layer glue. The Axon CNN canary lives
  in `mnist_cnn_full_test.exs` (`:training_full`).
  """

  use ExUnit.Case, async: true

  alias Emily.TrainingHelper, as: TH
  import TH, only: [close?: 4, flunk_trajectory: 5]

  @input_shape {1, 10, 10}
  @batch 4
  @classes 3
  @steps 30
  @lr_val 0.05

  test "per-step CNN loss trajectory matches BinaryBackend" do
    # BinaryBackend oracle.
    params_bin = TH.init_cnn(@input_shape, @classes, 0, Nx.BinaryBackend)
    {x_bin, y_bin} = TH.cnn_batch({@batch, 10, 10}, @classes, Nx.BinaryBackend)
    lr_bin = Nx.tensor(@lr_val, type: {:f, 32}, backend: Nx.BinaryBackend)

    losses_bin =
      TH.run_steps(
        &TH.cnn_step_with_loss/4,
        params_bin,
        [x_bin, y_bin, lr_bin],
        @steps,
        Nx.Defn.Evaluator
      )

    # Emily side — same init, same data, Emily.Compiler.
    params_emily = TH.init_cnn(@input_shape, @classes, 0, Emily.Backend)
    {x_emily, y_emily} = TH.cnn_batch({@batch, 10, 10}, @classes, Emily.Backend)
    lr_emily = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)

    losses_emily =
      TH.run_steps(
        &TH.cnn_step_with_loss/4,
        params_emily,
        [x_emily, y_emily, lr_emily],
        @steps,
        Emily.Compiler
      )

    assert length(losses_emily) == @steps
    assert length(losses_bin) == @steps

    # Per-step match — silent-drift canary. Looser than MLP because
    # the CNN accumulates through conv → maxpool → conv → maxpool and
    # each boundary contributes f32 reduction-order drift. 1e-2 rtol
    # still catches gross divergence from step ~5 onward.
    for {{le, lb}, i} <- Enum.zip(losses_emily, losses_bin) |> Enum.with_index() do
      close?(le, lb, 1.0e-4, 1.0e-2) ||
        flunk_trajectory(i, le, lb, losses_emily, losses_bin)
    end

    # Final loss should have decreased.
    assert List.first(losses_bin) > List.last(losses_bin)
    assert List.first(losses_emily) > List.last(losses_emily)

    le_final = List.last(losses_emily)
    lb_final = List.last(losses_bin)

    assert close?(le_final, lb_final, 1.0e-4, 1.0e-2),
           "final loss divergence: emily=#{le_final} bin=#{lb_final} " <>
             "reldiff=#{abs(le_final - lb_final) / abs(lb_final)}"
  end
end
