defmodule Emily.GradZoo do
  @moduledoc """
  Shared gradient test functions for the M9/M13 grad suites.

  Contains the `defn` functions that constitute the "grad zoo" — the
  canonical set of differentiable computations used across:

    * `grad_equivalence_test.exs` (M9 Phase B: property-based oracle)
    * `finite_diff_test.exs` (M9 Phase C: numerical oracle — uses
      `grad_sum_op` and `grad_dot_left` from here; keeps its own
      `grad_logsumexp` and `grad_sigmoid_sum`)
    * `exla_oracle_test.exs` (M13: EXLA golden-value conformance)
    * `bench/exla_golden_gen.exs` (M13: standalone golden generator
      script — maintains its own copy of these functions since it
      can't depend on Emily's `test/support/`)

  Each function takes BinaryBackend or Emily.Backend tensors and
  returns a gradient tensor via `Nx.Defn.grad`.

  `fixed_inputs/1` returns deterministic BinaryBackend tensors for each
  function — used by the EXLA golden generator and the conformance test.
  Property tests in `grad_equivalence_test.exs` continue to use
  StreamData generators for broader coverage.
  """

  import Nx.Defn

  # -------------------- Zoo functions --------------------

  defn(grad_sum_op(x), do: grad(x, fn z -> Nx.sum(z) end))

  defn(grad_dot_left(x, b), do: grad(x, fn z -> z |> Nx.dot(b) |> Nx.sum() end))

  defn grad_reshape_transpose(x) do
    grad(x, fn z ->
      z |> Nx.transpose(axes: [1, 0]) |> Nx.reshape({12}) |> Nx.sum()
    end)
  end

  defn grad_broadcast(x) do
    grad(x, fn z -> z |> Nx.broadcast({4, 3}) |> Nx.sum() end)
  end

  defn grad_gather(x, idx) do
    grad(x, fn z -> z |> Nx.gather(idx, axes: [0, 1]) |> Nx.sum() end)
  end

  defn grad_indexed_add(x, idx, upd) do
    grad(x, fn z -> z |> Nx.indexed_add(idx, upd) |> Nx.sum() end)
  end

  defn grad_gather_dot_softmax(x, idx, w) do
    grad(x, fn z ->
      z
      |> Nx.gather(idx, axes: [0])
      |> Nx.reshape({3, 6})
      |> Nx.dot(w)
      |> softmax_last()
      |> Nx.sum()
    end)
  end

  defn grad_attention(x, wq, wk, wv, scale) do
    grad(x, fn z ->
      q = Nx.dot(z, wq)
      k = Nx.dot(z, wk)
      v = Nx.dot(z, wv)
      logits = Nx.dot(q, Nx.transpose(k)) * scale
      attn = softmax_last(logits)
      attn |> Nx.dot(v) |> Nx.sum()
    end)
  end

  defn softmax_last(t) do
    m = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    e = Nx.exp(t - m)
    e / Nx.sum(e, axes: [-1], keep_axes: true)
  end

  # -------------------- Fixed inputs --------------------

  @doc """
  Returns a list of deterministic `Nx.BinaryBackend` tensors suitable as
  arguments to the named zoo function. Used by the EXLA golden generator
  and the M13 conformance test.

  Inputs are built via the same sin-based `det_weights` pattern as
  `TrainingHelper` — same seed produces bit-identical values on any
  backend.
  """
  def fixed_inputs(:grad_sum_op) do
    [det_weights({3, 4}, 1)]
  end

  def fixed_inputs(:grad_dot_left) do
    [det_weights({3, 4}, 2), det_weights({4, 5}, 3)]
  end

  def fixed_inputs(:grad_reshape_transpose) do
    [det_weights({3, 4}, 4)]
  end

  def fixed_inputs(:grad_broadcast) do
    [det_weights({3}, 5)]
  end

  def fixed_inputs(:grad_gather) do
    [
      det_weights({4, 5}, 6),
      Nx.tensor([[0, 1], [2, 3], [1, 0]], type: {:s, 32}, backend: Nx.BinaryBackend)
    ]
  end

  def fixed_inputs(:grad_indexed_add) do
    [
      det_weights({3, 4}, 7),
      Nx.tensor([[0, 1], [2, 3], [1, 0]], type: {:s, 32}, backend: Nx.BinaryBackend),
      Nx.iota({3}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.add(1.0)
    ]
  end

  def fixed_inputs(:grad_gather_dot_softmax) do
    [
      det_weights({4, 6}, 8),
      Nx.tensor([[0], [2], [1]], backend: Nx.BinaryBackend),
      Nx.iota({6, 5}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(30.0)
    ]
  end

  def fixed_inputs(:grad_attention) do
    [
      det_weights({3, 4}, 9),
      Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0),
      Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0),
      Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0),
      Nx.tensor(0.5, type: {:f, 32}, backend: Nx.BinaryBackend)
    ]
  end

  @doc """
  Returns the function capture for the named zoo function.
  """
  def grad_function(:grad_sum_op), do: &grad_sum_op/1
  def grad_function(:grad_dot_left), do: &grad_dot_left/2
  def grad_function(:grad_reshape_transpose), do: &grad_reshape_transpose/1
  def grad_function(:grad_broadcast), do: &grad_broadcast/1
  def grad_function(:grad_gather), do: &grad_gather/2
  def grad_function(:grad_indexed_add), do: &grad_indexed_add/3
  def grad_function(:grad_gather_dot_softmax), do: &grad_gather_dot_softmax/3
  def grad_function(:grad_attention), do: &grad_attention/5

  @doc "All zoo function identifiers, in order."
  def all_functions do
    [
      :grad_sum_op,
      :grad_dot_left,
      :grad_reshape_transpose,
      :grad_broadcast,
      :grad_gather,
      :grad_indexed_add,
      :grad_gather_dot_softmax,
      :grad_attention
    ]
  end

  # -------------------- Internal --------------------

  # Same deterministic init as TrainingHelper.det_weights/3 but always
  # on BinaryBackend (the conformance test transfers to Emily itself).
  defp det_weights(shape, seed) do
    size = shape |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

    Nx.iota({size}, type: {:f, 32}, backend: Nx.BinaryBackend)
    |> Nx.multiply(0.7)
    |> Nx.add(seed * 7.1)
    |> Nx.sin()
    |> Nx.multiply(0.3)
    |> Nx.reshape(shape)
  end
end
