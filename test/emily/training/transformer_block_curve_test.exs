defmodule Emily.Training.TransformerBlockCurveTest do
  @moduledoc """
  Training-curve matching for a handwritten single transformer block
  (M9 Phase D).

  Architecture: single-head self-attention (Q/K/V + output projection)
  + FFN (2-layer ReLU) + residuals on both sub-blocks. No layer-norm —
  keeps the grad surface elementary enough that a curve divergence
  points at specific backend ops rather than LN's sum+rsqrt stack.

  Exercises the attention-shaped backward path that Phase B's
  composition case hits only superficially. Same per-step and
  final-loss tolerances as `mlp_curve_test.exs`.
  """

  use ExUnit.Case, async: true

  alias Emily.TrainingHelper, as: TH
  import TH, only: [close?: 4, flunk_trajectory: 5]

  @embed 16
  @ff 32
  @seq 8
  @steps 50
  @lr_val 0.1

  test "per-step and final loss match BinaryBackend" do
    scale_val = 1.0 / :math.sqrt(@embed)

    # BinaryBackend side.
    params_bin = TH.init_block({@embed, @ff}, 0, Nx.BinaryBackend)
    {x_bin, y_bin} = TH.block_batch({@seq, @embed}, Nx.BinaryBackend)
    lr_bin = Nx.tensor(@lr_val, type: {:f, 32}, backend: Nx.BinaryBackend)
    scale_bin = Nx.tensor(scale_val, type: {:f, 32}, backend: Nx.BinaryBackend)

    losses_bin =
      TH.run_steps(
        &TH.block_step_with_loss/5,
        params_bin,
        [x_bin, y_bin, lr_bin, scale_bin],
        @steps,
        Nx.Defn.Evaluator
      )

    # Emily side.
    params_emily = TH.init_block({@embed, @ff}, 0, Emily.Backend)
    {x_emily, y_emily} = TH.block_batch({@seq, @embed}, Emily.Backend)
    lr_emily = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)
    scale_emily = Nx.tensor(scale_val, type: {:f, 32}, backend: Emily.Backend)

    losses_emily =
      TH.run_steps(
        &TH.block_step_with_loss/5,
        params_emily,
        [x_emily, y_emily, lr_emily, scale_emily],
        @steps,
        Emily.Compiler
      )

    # Sanity: loss decreased overall.
    assert List.first(losses_bin) > List.last(losses_bin),
           "BinaryBackend training didn't reduce loss: " <>
             "first=#{List.first(losses_bin)} last=#{List.last(losses_bin)}"

    # Per-step match (rtol=1e-3 drift canary).
    for {{le, lb}, i} <- Enum.zip(losses_emily, losses_bin) |> Enum.with_index() do
      close?(le, lb, 1.0e-4, 1.0e-3) ||
        flunk_trajectory(i, le, lb, losses_emily, losses_bin)
    end

    # Final-loss match (rtol=1e-4 convergence correctness).
    le_final = List.last(losses_emily)
    lb_final = List.last(losses_bin)

    assert close?(le_final, lb_final, 1.0e-5, 1.0e-4),
           "final loss divergence: emily=#{le_final} bin=#{lb_final} " <>
             "reldiff=#{abs(le_final - lb_final) / abs(lb_final)}"
  end
end
