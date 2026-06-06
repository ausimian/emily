defmodule Emily.Training.CnnNativeCurveTest do
  @moduledoc """
  Native single-NIF CNN training convergence (issue #174).

  The training analogue of the conformance native lanes
  (`Emily.Conformance.CompilerNativeTest`): a full conv + maxpool
  training step — forward, backward, grad, and SGD update — is driven
  through `compiler: Emily.Compiler, native: true, native_fallback:
  :raise` for 30 steps and the per-step loss trajectory is checked
  against two references.

  Why this exists. `cnn_curve_test.exs` already curve-matches the
  handwritten CNN, but only in **eval** mode (`Emily.Compiler` walking
  the Expr op-by-op via the Evaluator). Every other `training/*` test
  is eval-only too, so CNN training was verified-lowering (the
  `compiler_equivalence_test.exs` op gates) but never **convergence**-
  tested under the single-NIF replay. This closes that gap.

  Three lanes, same deterministic init and data:

    * **native** — `native: true, native_fallback: :raise`. The
      `:raise` makes this a no-fallback gate: if any op in the
      forward+backward+grad+SGD step fails to lower (the maxpool
      backward lands on `window_scatter_max` every step; the conv
      backward flips the kernel with `reverse`), the run raises here
      instead of silently degrading to the evaluator.
    * **eval** — `Emily.Compiler` op-by-op. Same MLX kernels in the
      same order as the native replay, so the two track **bit-
      identically** through training. A 1e-6 bar asserts the single-
      NIF lowering reproduces op-by-op exactly across 30 SGD updates.
    * **binary** — `Nx.Defn.Evaluator` on `Nx.BinaryBackend`, the
      non-MLX convergence oracle. Looser bar (1e-2 rtol, as in
      `cnn_curve_test.exs`) absorbs f32 reduction-order drift between
      MLX's parallel reductions and BinaryBackend's sequential ones.

  No Axon — the handwritten path keeps the failure surface tiny (see
  `cnn_curve_test.exs`). The Axon CNN canary stays in
  `mnist_cnn_full_test.exs` (`:training_full`).
  """

  use ExUnit.Case, async: true

  alias Emily.TrainingHelper, as: TH
  import TH, only: [close?: 4, flunk_trajectory: 5]

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]

  @input_shape {1, 10, 10}
  @batch 4
  @classes 3
  @steps 30
  @lr_val 0.05

  test "per-step CNN loss trajectory matches under native single-NIF compile" do
    # Native single-NIF lane — the system under test. `native_fallback:
    # :raise` proves full native coverage of the training step.
    params_native = TH.init_cnn(@input_shape, @classes, 0, Emily.Backend)
    {x_native, y_native} = TH.cnn_batch({@batch, 10, 10}, @classes, Emily.Backend)
    lr_native = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)

    losses_native =
      TH.run_steps(
        &TH.cnn_step_with_loss/4,
        params_native,
        [x_native, y_native, lr_native],
        @steps,
        @native
      )

    # Op-by-op Emily eval lane — same MLX kernels, isolates single-NIF
    # lowering bugs from backend numerics.
    params_eval = TH.init_cnn(@input_shape, @classes, 0, Emily.Backend)
    {x_eval, y_eval} = TH.cnn_batch({@batch, 10, 10}, @classes, Emily.Backend)
    lr_eval = Nx.tensor(@lr_val, type: {:f, 32}, backend: Emily.Backend)

    losses_eval =
      TH.run_steps(
        &TH.cnn_step_with_loss/4,
        params_eval,
        [x_eval, y_eval, lr_eval],
        @steps,
        @eval
      )

    # BinaryBackend oracle — the non-MLX convergence reference.
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

    assert length(losses_native) == @steps
    assert length(losses_eval) == @steps
    assert length(losses_bin) == @steps

    # 1. Single-NIF native == op-by-op eval. Both are MLX in the same
    #    order, so they track bit-identically; the tight bar makes a
    #    divergent native trajectory a hard failure.
    for {{ln, le}, i} <- Enum.zip(losses_native, losses_eval) |> Enum.with_index() do
      close?(ln, le, 1.0e-6, 1.0e-6) ||
        flunk_trajectory(i, ln, le, losses_native, losses_eval)
    end

    # 2. Native trajectory matches the BinaryBackend oracle within the
    #    CNN tolerance — same bar as cnn_curve_test.exs.
    for {{ln, lb}, i} <- Enum.zip(losses_native, losses_bin) |> Enum.with_index() do
      close?(ln, lb, 1.0e-4, 1.0e-2) ||
        flunk_trajectory(i, ln, lb, losses_native, losses_bin)
    end

    # 3. Convergence — the native loss actually decreased over the run.
    assert List.first(losses_native) > List.last(losses_native),
           "native loss did not decrease: first=#{List.first(losses_native)} " <>
             "last=#{List.last(losses_native)}"

    # 4. Final loss agrees with the oracle (convergence correctness:
    #    catches a run where per-step drift averaged out but the
    #    optimizer ended up somewhere wrong).
    ln_final = List.last(losses_native)
    lb_final = List.last(losses_bin)

    assert close?(ln_final, lb_final, 1.0e-4, 1.0e-2),
           "final loss divergence: native=#{ln_final} bin=#{lb_final} " <>
             "reldiff=#{abs(ln_final - lb_final) / abs(lb_final)}"
  end
end
