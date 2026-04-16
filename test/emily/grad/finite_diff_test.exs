defmodule Emily.Grad.FiniteDiffTest do
  @moduledoc """
  Finite-difference numerical-gradient oracle (M9 Phase C).

  Pilot scope: four structurally-distinct ops — `sum`, `dot`,
  `logsumexp` (= log(sum(exp(x))), whose grad is softmax), and
  `sigmoid`. For each, compute the symbolic gradient under
  `Emily.Compiler` and compare against the numerical central-
  difference gradient with per-op tolerance.

  Deliberately small. The harness is the load-bearing artefact; the
  op set is not meant to be comprehensive. If the harness proves
  itself during M9, future milestones can expand the zoo.
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [to_emily: 1]
  import Emily.GradZoo, only: [grad_sum_op: 1, grad_dot_left: 2]
  import Nx.Defn

  alias Emily.GradHelper

  # grad_sum_op/1 and grad_dot_left/2 live in Emily.GradZoo (shared
  # with the equivalence and EXLA oracle suites). The two below are
  # finite-diff-only pilots not in the cross-backend zoo.

  defn(grad_logsumexp(x), do: grad(x, fn z -> z |> Nx.exp() |> Nx.sum() |> Nx.log() end))

  defn(grad_sigmoid_sum(x), do: grad(x, fn z -> z |> Nx.sigmoid() |> Nx.sum() end))

  # ---- Test cases ----

  describe "sum" do
    test "symbolic grad matches finite-difference" do
      x = Nx.tensor([[1.0, -2.0, 0.5], [0.25, -0.75, 1.5]], backend: Nx.BinaryBackend)

      sym =
        Nx.Defn.jit_apply(&grad_sum_op/1, [to_emily(x)], compiler: Emily.Compiler)

      num = GradHelper.finite_diff(fn z -> Nx.sum(z) end, x)

      GradHelper.assert_grad_close(sym, num, :sum)
    end
  end

  describe "dot" do
    test "grad w.r.t. left operand matches finite-difference" do
      x = Nx.tensor([[0.5, -1.0, 0.25], [1.5, 0.0, -0.5]], backend: Nx.BinaryBackend)

      b =
        Nx.tensor(
          [
            [1.0, 0.5],
            [-0.25, 0.75],
            [0.5, -1.0]
          ],
          backend: Nx.BinaryBackend
        )

      sym =
        Nx.Defn.jit_apply(
          &grad_dot_left/2,
          [to_emily(x), to_emily(b)],
          compiler: Emily.Compiler
        )

      # FD closes over the BinaryBackend `b` — don't use Emily tensors
      # in the forward closure (perturbed x is on BinaryBackend).
      num = GradHelper.finite_diff(fn z -> z |> Nx.dot(b) |> Nx.sum() end, x)

      GradHelper.assert_grad_close(sym, num, :dot)
    end
  end

  describe "logsumexp (grad = softmax)" do
    # log(sum(exp(x))) — well-defined loss, gradient is softmax(x).
    # Non-trivial, well-conditioned for small magnitudes. Sum-of-
    # softmax ≡ 1 so differentiating `sum(softmax(x))` directly would
    # give identically-zero gradients (degenerate test).
    test "grad matches finite-difference" do
      x =
        Nx.tensor([[0.3, -0.5, 1.2, -0.7], [0.1, 0.4, -1.1, 0.6]], backend: Nx.BinaryBackend)

      sym =
        Nx.Defn.jit_apply(&grad_logsumexp/1, [to_emily(x)], compiler: Emily.Compiler)

      num = GradHelper.finite_diff(fn z -> z |> Nx.exp() |> Nx.sum() |> Nx.log() end, x)

      GradHelper.assert_grad_close(sym, num, :logsumexp)
    end
  end

  describe "sigmoid" do
    test "grad matches finite-difference" do
      x =
        Nx.tensor([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]], backend: Nx.BinaryBackend)

      sym =
        Nx.Defn.jit_apply(&grad_sigmoid_sum/1, [to_emily(x)], compiler: Emily.Compiler)

      num = GradHelper.finite_diff(fn z -> z |> Nx.sigmoid() |> Nx.sum() end, x)

      GradHelper.assert_grad_close(sym, num, :sigmoid)
    end
  end
end
