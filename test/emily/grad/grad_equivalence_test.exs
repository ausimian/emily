defmodule Emily.Grad.EquivalenceTest do
  @moduledoc """
  Grad-equivalence oracle tests. For each op in the zoo, assert that
  `grad` computed under `compiler: Emily.Compiler` matches the same
  grad computed under `compiler: Nx.Defn.Evaluator` on BinaryBackend,
  within a grad-scaled tolerance.

  Grad is symbolic in Elixir — it lowers to the same forward ops the
  backend implements. The structural correctness of each grad rule is
  Nx's responsibility; what we verify here is that every op the
  backward walk lands on produces the same numerics as BinaryBackend,
  AND that the Emily.Compiler JIT path handles the grad walk (rather
  than just the `Nx.Defn.Evaluator` path).

  Zoo scope is deliberately narrow (~6 ops + 2 compositions): M2
  already covers forward correctness per op. Grad equivalence is a
  structural property — once the walk reaches the right backend ops
  for a handful of grad-rule shapes, coverage is effectively complete.

  Every case goes through `Nx.Defn.jit_apply` because `Nx.Defn.grad`
  outside `defn` can't convert Emily.Backend tensors captured in
  closures to expressions. Passing tensors as `defn` arguments avoids
  the capture and — usefully — exercises the Compiler path as a
  side-effect, unifying what PLAN.md labelled B and B.5.

  The final `describe "PRNG-key threading"` block covers B.6.
  """

  use ExUnit.Case, async: true
  use ExUnitProperties

  import Emily.BackendGenerators
  import Emily.GradZoo
  import Nx.Defn

  @max_runs 10

  # Grad magnitudes can exceed input magnitudes (chain-rule amplifies),
  # so loosen the M2 float tolerance by ~10× for grad comparisons.
  defp grad_tol({:f, _}), do: 1.0e-3
  defp grad_tol(_), do: 0.0

  # Run the same defn twice: once through Emily.Compiler on
  # Emily-backed inputs, once through Nx.Defn.Evaluator on
  # BinaryBackend inputs. Return {emily, ref}.
  defp through_both(fun, args) do
    emily_args = Enum.map(args, &to_emily/1)

    emily = Nx.Defn.jit_apply(fun, emily_args, compiler: Emily.Compiler)
    ref = Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)

    {emily, ref}
  end

  # ---------------- Zoo ----------------
  # defn functions live in Emily.GradZoo (shared with the M13 EXLA
  # oracle test and the golden generator).

  describe "sum" do
    property "grad is ones-shaped like input" do
      check all(
              shape <- non_scalar_shape(),
              x <- tensor(shape, {:f, 32}),
              max_runs: @max_runs
            ) do
        {emily, ref} = through_both(&grad_sum_op/1, [x])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "dot" do
    property "grad w.r.t. left operand matches" do
      check all(
              m <- StreamData.integer(1..5),
              k <- StreamData.integer(1..5),
              n <- StreamData.integer(1..5),
              a <- tensor({m, k}, {:f, 32}),
              b <- tensor({k, n}, {:f, 32}),
              max_runs: @max_runs
            ) do
        {emily, ref} = through_both(&grad_dot_left/2, [a, b])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "reshape ∘ transpose" do
    property "grad passes through shape ops" do
      check all(x <- tensor({3, 4}, {:f, 32}), max_runs: @max_runs) do
        {emily, ref} = through_both(&grad_reshape_transpose/1, [x])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "broadcast" do
    property "grad sums over broadcasted dims" do
      check all(x <- tensor({3}, {:f, 32}), max_runs: @max_runs) do
        {emily, ref} = through_both(&grad_broadcast/1, [x])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "gather" do
    # Grad of gather lands on indexed_add — exercises the new native
    # scatter_add path through the backward walk.
    property "multi-axis gather grad scatter-adds to the gathered positions" do
      check all(
              x <- tensor({4, 5}, {:f, 32}),
              idx <- unique_index_tensor(3, [0..3, 0..4]),
              max_runs: @max_runs
            ) do
        {emily, ref} = through_both(&grad_gather/2, [x, idx])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "indexed_add" do
    property "grad routes updates-grad and target-grad" do
      check all(
              x <- tensor({3, 4}, {:f, 32}),
              idx <- unique_index_tensor(3, [0..2, 0..3]),
              max_runs: @max_runs
            ) do
        n = elem(idx.shape, 0)

        upd =
          Nx.iota({n}, type: {:f, 32}, backend: Nx.BinaryBackend)
          |> Nx.add(1.0)

        {emily, ref} = through_both(&grad_indexed_add/3, [x, idx, upd])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  # ---------------- Composition ----------------

  describe "composition: gather → dot → softmax → sum" do
    property "stacked grad matches" do
      check all(x <- tensor({4, 6}, {:f, 32}), max_runs: @max_runs) do
        idx = Nx.tensor([[0], [2], [1]], backend: Nx.BinaryBackend)

        w =
          Nx.iota({6, 5}, type: {:f, 32}, backend: Nx.BinaryBackend)
          |> Nx.divide(30.0)

        {emily, ref} = through_both(&grad_gather_dot_softmax/3, [x, idx, w])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  # ---------------- Window ops (M17) ----------------

  describe "window_sum grad" do
    property "grad matches BinaryBackend" do
      check all(x <- tensor({2, 3, 4, 4}, {:f, 32}), max_runs: @max_runs) do
        {emily, ref} = through_both(&grad_window_sum/1, [x])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "window_max grad — lands on window_scatter_max" do
    # Nx's grad rule rewrites grad(window_max) into window_scatter_max.
    # This property test is the structural proof that the whole
    # grad-of-maxpool chain (pad → as_strided → argmax → scatter_add)
    # is numerically equivalent to the BinaryBackend reference.
    property "grad matches BinaryBackend" do
      check all(x <- tensor({2, 3, 4, 4}, {:f, 32}), max_runs: @max_runs) do
        {emily, ref} = through_both(&grad_window_max_pool/1, [x])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "window_avg grad (sum / kernel_size)" do
    property "grad matches BinaryBackend" do
      check all(x <- tensor({2, 3, 4, 4}, {:f, 32}), max_runs: @max_runs) do
        {emily, ref} = through_both(&grad_window_avg_pool/1, [x])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  describe "composition: attention-shaped block" do
    # Mini attention: Q/K/V projections, scaled dot-product, softmax,
    # output projection, sum. Grad w.r.t. input x exercises dot,
    # transpose, softmax composition, elementwise scaling, and the
    # broadcast/reduction grad rules in combination.
    property "grad matches BinaryBackend" do
      check all(x <- tensor({3, 4}, {:f, 32}), max_runs: @max_runs) do
        wq = Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0)
        wk = Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0)
        wv = Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0)
        scale = Nx.tensor(0.5, type: {:f, 32}, backend: Nx.BinaryBackend)

        {emily, ref} = through_both(&grad_attention/5, [x, wq, wk, wv, scale])
        assert_close(emily, ref, tol: grad_tol({:f, 32}))
      end
    end
  end

  # ---------------- PRNG-key threading (B.6) ----------------
  #
  # A `defn` calling `Nx.Random.uniform_split` inside the computation
  # being differentiated — i.e. grad through stochastic dropout.
  # Running it twice with the same key must produce bit-identical
  # output (determinism). The full shape/value match against the
  # Evaluator path is deferred: MLX's threefry kernel and Nx's
  # reference threefry can produce different mask bit patterns even
  # from the same seed, so the grad's nonzero pattern differs. What we
  # can assert cross-backend is that grads of the same shape come out;
  # within Emily, same-key runs are bit-identical.

  defn grad_dropout(x, key) do
    grad(x, fn z ->
      mask = Nx.Random.uniform_split(key, 0.0, 1.0, shape: Nx.shape(z))
      kept = Nx.select(Nx.greater(mask, 0.5), z, Nx.tensor(0.0, type: Nx.type(z)))
      Nx.sum(kept)
    end)
  end

  describe "PRNG-key threading" do
    test "dropout grad under Emily.Compiler is deterministic, matches shape of Evaluator" do
      x =
        Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend)
        |> Nx.divide(16.0)

      key = Nx.Random.key(42)

      emily_1 =
        Nx.Defn.jit_apply(
          &grad_dropout/2,
          [to_emily(x), to_emily(key)],
          compiler: Emily.Compiler
        )

      emily_2 =
        Nx.Defn.jit_apply(
          &grad_dropout/2,
          [to_emily(x), to_emily(key)],
          compiler: Emily.Compiler
        )

      ref =
        Nx.Defn.jit_apply(&grad_dropout/2, [x, key], compiler: Nx.Defn.Evaluator)

      # Same key → bit-identical output across runs: the grad walk
      # threads the key deterministically through the compiled
      # function.
      assert_close(emily_1, emily_2, tol: 0.0)

      # Shape agreement with Evaluator: the compiled grad produces a
      # correctly-shaped output. Value comparison isn't meaningful
      # cross-backend because the RNG mask bits may differ.
      assert emily_1.shape == ref.shape
      assert emily_1.type == ref.type
    end
  end

  # ---------------- Generators ----------------

  # Generate a {:s, 32} index tensor of shape {n, length(ranges)} where
  # rows are all unique (so MLX scatter and Nx.indexed_put agree on
  # duplicates). `ranges` is a list of integer ranges, one per axis.
  defp unique_index_tensor(max_n, ranges) do
    dim_generators = Enum.map(ranges, &StreamData.integer/1)

    StreamData.list_of(StreamData.fixed_list(dim_generators),
      min_length: 1,
      max_length: max_n
    )
    |> StreamData.map(&Enum.uniq/1)
    |> StreamData.filter(&(&1 != []))
    |> StreamData.map(fn rows ->
      Nx.tensor(rows, type: {:s, 32}, backend: Nx.BinaryBackend)
    end)
  end
end
