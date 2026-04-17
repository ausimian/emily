defmodule Emily.Training.Bf16MlpCurveTest do
  @moduledoc """
  Mixed-precision MLP curve-matching (M16).

  Same structure as `MlpCurveTest`, but exercises the full
  mixed-precision recipe: f32 master weights, bf16 inputs, loss
  scaling inside the grad closure, and f32 gradient accumulation.

  Tolerances are wider than the f32 test (rtol 5e-2 vs 1e-3) because
  bf16 arithmetic has ~3 decimal digits of precision.
  """

  use ExUnit.Case, async: true

  alias Emily.TrainingHelper, as: TH
  import TH, only: [close?: 4, flunk_trajectory: 5]

  @dims {4, 8, 3}
  @batch_shape {16, 4, 3}
  @steps 50
  @lr_val 0.5
  @loss_scale_val 256.0

  test "bf16 mixed-precision loss trajectory matches BinaryBackend within tolerance" do
    # BinaryBackend oracle — f32 master params, bf16 inputs.
    params_bin = TH.init_mlp(@dims, 0, Nx.BinaryBackend)
    {x_bin, y_bin} = TH.mlp_batch(@batch_shape, Nx.BinaryBackend)
    x_bin = Nx.as_type(x_bin, {:bf, 16})
    y_bin = Nx.as_type(y_bin, {:bf, 16})
    lr_bin = Nx.tensor(@lr_val, type: {:f, 32}, backend: Nx.BinaryBackend)
    scale_bin = Nx.tensor(@loss_scale_val, type: {:f, 32}, backend: Nx.BinaryBackend)

    losses_bin =
      TH.run_steps(
        &TH.mlp_mp_step_with_loss/5,
        params_bin,
        [x_bin, y_bin, lr_bin, scale_bin],
        @steps,
        Nx.Defn.Evaluator
      )

    # Emily side — bit-identical starting point, compiled via Emily.Compiler.
    params_emily = TH.init_mlp(@dims, 0, Emily.Backend)
    {x_emily, y_emily} = TH.mlp_batch(@batch_shape, Emily.Backend)
    x_emily = Nx.as_type(x_emily, {:bf, 16})
    y_emily = Nx.as_type(y_emily, {:bf, 16})
    lr_emily = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)
    scale_emily = Nx.tensor(@loss_scale_val, type: {:f, 32}, backend: Emily.Backend)

    losses_emily =
      TH.run_steps(
        &TH.mlp_mp_step_with_loss/5,
        params_emily,
        [x_emily, y_emily, lr_emily, scale_emily],
        @steps,
        Emily.Compiler
      )

    assert length(losses_emily) == @steps
    assert length(losses_bin) == @steps

    # Loss decreased over the run.
    assert List.first(losses_bin) > List.last(losses_bin)

    # Per-step match — bf16-appropriate tolerance.
    for {{le, lb}, i} <- Enum.zip(losses_emily, losses_bin) |> Enum.with_index() do
      close?(le, lb, 1.0e-2, 5.0e-2) ||
        flunk_trajectory(i, le, lb, losses_emily, losses_bin)
    end

    # Final-loss match.
    le_final = List.last(losses_emily)
    lb_final = List.last(losses_bin)

    assert close?(le_final, lb_final, 1.0e-2, 5.0e-2),
           "final loss divergence: emily=#{le_final} bin=#{lb_final} " <>
             "reldiff=#{abs(le_final - lb_final) / abs(lb_final)}"
  end
end
