defmodule Emily.Training.MlpNativeCurveTest do
  @moduledoc """
  Native single-NIF MLP training convergence (issue #174).

  The dense + SGD companion to `cnn_native_curve_test.exs`: a 2-layer
  ReLU MLP training step — forward, backward, grad, SGD update — is
  driven through `compiler: Emily.Compiler, native: true,
  native_fallback: :raise` for 50 steps and the per-step loss
  trajectory is checked against two references.

  This closes the matmul-dominated half of the training-coverage gap
  the issue calls out: `mlp_curve_test.exs` already curve-matches this
  MLP, but only in eval mode. Here the same step replays through the
  single NIF, with `:raise` proving the dense forward/backward and SGD
  update lower with **zero** fallback.

  Three lanes, same deterministic init and data:

    * **native** — single-NIF replay, no-fallback gate.
    * **eval** — `Emily.Compiler` op-by-op; bit-identical to native
      (same MLX kernels, same order), asserted at a 1e-6 bar.
    * **binary** — `Nx.Defn.Evaluator` on `Nx.BinaryBackend`, the
      non-MLX oracle. The MLP is matmul-dominated, so the bar matches
      `mlp_curve_test.exs` (1e-3 per-step rtol, 1e-4 final).
  """

  use ExUnit.Case, async: true

  alias Emily.TrainingHelper, as: TH
  import TH, only: [close?: 4, flunk_trajectory: 5]

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]

  @dims {4, 8, 3}
  @batch_shape {16, 4, 3}
  @steps 50
  @lr_val 0.5

  test "per-step MLP loss trajectory matches under native single-NIF compile" do
    # Native single-NIF lane — the system under test.
    params_native = TH.init_mlp(@dims, 0, Emily.Backend)
    {x_native, y_native} = TH.mlp_batch(@batch_shape, Emily.Backend)
    lr_native = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)

    losses_native =
      TH.run_steps(
        &TH.mlp_step_with_loss/4,
        params_native,
        [x_native, y_native, lr_native],
        @steps,
        @native
      )

    # Op-by-op Emily eval lane — same MLX kernels.
    params_eval = TH.init_mlp(@dims, 0, Emily.Backend)
    {x_eval, y_eval} = TH.mlp_batch(@batch_shape, Emily.Backend)
    lr_eval = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)

    losses_eval =
      TH.run_steps(
        &TH.mlp_step_with_loss/4,
        params_eval,
        [x_eval, y_eval, lr_eval],
        @steps,
        @eval
      )

    # BinaryBackend oracle.
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

    assert length(losses_native) == @steps
    assert length(losses_eval) == @steps
    assert length(losses_bin) == @steps

    # 1. Single-NIF native == op-by-op eval (bit-identical MLX path).
    for {{ln, le}, i} <- Enum.zip(losses_native, losses_eval) |> Enum.with_index() do
      close?(ln, le, 1.0e-6, 1.0e-6) ||
        flunk_trajectory(i, ln, le, losses_native, losses_eval)
    end

    # 2. Native trajectory matches the BinaryBackend oracle — same bar
    #    as mlp_curve_test.exs.
    for {{ln, lb}, i} <- Enum.zip(losses_native, losses_bin) |> Enum.with_index() do
      close?(ln, lb, 1.0e-4, 1.0e-3) ||
        flunk_trajectory(i, ln, lb, losses_native, losses_bin)
    end

    # 3. Convergence — the native loss actually decreased over the run.
    assert List.first(losses_native) > List.last(losses_native),
           "native loss did not decrease: first=#{List.first(losses_native)} " <>
             "last=#{List.last(losses_native)}"

    # 4. Final loss agrees with the oracle (convergence correctness).
    ln_final = List.last(losses_native)
    lb_final = List.last(losses_bin)

    assert close?(ln_final, lb_final, 1.0e-5, 1.0e-4),
           "final loss divergence: native=#{ln_final} bin=#{lb_final} " <>
             "reldiff=#{abs(ln_final - lb_final) / abs(lb_final)}"
  end
end
