defmodule Emily.GradHelper do
  @moduledoc """
  Finite-difference numerical-gradient oracle for M9 Phase C.

  Catches the class of grad bug where the symbolic path on Emily and
  on `Nx.BinaryBackend` *agree* but are both wrong (e.g. a shared
  misinterpretation of a forward op's semantics). Works by computing
  the gradient via central differences — a totally independent path
  that only uses forward evaluation.

  Deliberately narrow (~4 pilot ops): f32 central differences bottom
  out around 1e-3 relative, so the tolerance per op has to be
  calibrated empirically. Expanding the op set is out of M9 scope.

  All finite differences happen on `Nx.BinaryBackend` for exact scalar
  access; the symbolic grad runs on whichever backend the caller
  chose, and is transferred to BinaryBackend for comparison.
  """

  @doc """
  Central-difference numerical gradient of `fun` at `x`.

  `fun` must accept a tensor of the same shape as `x` and return a
  scalar tensor. Returns a tensor of the same shape as `x`.

  Pass `fun` closures over `Nx.BinaryBackend`-resident constants (e.g.
  a fixed weight matrix for a dot test) — perturbed inputs are built
  on `Nx.BinaryBackend`, so a capture of an Emily-backed tensor would
  cause a mixed-backend crash in the forward pass.

  Step size `eps` defaults to `1.0e-3`, empirically the sweet spot
  for f32 (smaller loses too much precision to cancellation; larger
  introduces too much Taylor-remainder error).
  """
  def finite_diff(fun, x, eps \\ 1.0e-3) do
    x_bin = Nx.backend_transfer(x, Nx.BinaryBackend)
    shape = x_bin.shape
    flat = Nx.to_flat_list(x_bin)
    size = length(flat)

    grads =
      for i <- 0..(size - 1) do
        plus = rebuild(List.update_at(flat, i, &(&1 + eps)), shape)
        minus = rebuild(List.update_at(flat, i, &(&1 - eps)), shape)
        fp = fun.(plus) |> Nx.to_number()
        fm = fun.(minus) |> Nx.to_number()
        (fp - fm) / (2 * eps)
      end

    rebuild(grads, shape)
  end

  @doc """
  Assert that a symbolic gradient matches a finite-difference gradient
  within the per-op tolerance.

  Tolerance tables are calibrated empirically: tight for linear ops
  (sum), looser for nonlinear (softmax, sigmoid) where f32 central
  differences bottom out around 1e-3 relative.
  """
  def assert_grad_close(symbolic, numerical, op, opts \\ []) do
    {atol, rtol} = tolerance_for(op)
    atol = opts[:atol] || atol
    rtol = opts[:rtol] || rtol

    sym_bin = Nx.backend_transfer(symbolic, Nx.BinaryBackend)
    num_bin = Nx.backend_transfer(numerical, Nx.BinaryBackend)

    if sym_bin.shape != num_bin.shape do
      ExUnit.Assertions.flunk(
        "grad shape mismatch for #{inspect(op)}: #{inspect(sym_bin.shape)} vs #{inspect(num_bin.shape)}"
      )
    end

    sym_list = Nx.to_flat_list(sym_bin)
    num_list = Nx.to_flat_list(num_bin)

    pairs = Enum.zip(sym_list, num_list) |> Enum.with_index()

    mismatches =
      pairs
      |> Enum.reject(fn {{s, n}, _i} -> close?(s, n, atol, rtol) end)
      |> Enum.take(5)

    if mismatches != [] do
      details =
        Enum.map_join(mismatches, "\n  ", fn {{s, n}, i} ->
          "[#{i}] symbolic=#{inspect(s)} numerical=#{inspect(n)} diff=#{inspect(abs(s - n))}"
        end)

      ExUnit.Assertions.flunk("""
      grad mismatch for #{inspect(op)} (atol=#{atol}, rtol=#{rtol}); first mismatches:
        #{details}
      """)
    end

    :ok
  end

  # --- Per-op tolerance tables ---
  #
  # `atol` — absolute tolerance (baseline noise floor).
  # `rtol` — relative tolerance (scaling term against |numerical|).
  #
  # `close?` compares: abs(s - n) <= atol + rtol * abs(n).
  # f32 central differences exhibit ~1e-5 absolute noise on sum-style
  # grads even in the linear case: the ulp at `sum(x_perturbed)` is
  # ~5.7e-7, scaled up by 1/(2*eps) ≈ 500× during the FD division.
  # :sum tolerance reflects that floor, not a looser symbolic bar.
  defp tolerance_for(:sum), do: {5.0e-5, 1.0e-4}
  defp tolerance_for(:dot), do: {1.0e-3, 1.0e-2}
  defp tolerance_for(:logsumexp), do: {2.0e-3, 2.0e-2}
  defp tolerance_for(:sigmoid), do: {1.0e-3, 1.0e-2}
  defp tolerance_for(_other), do: {1.0e-3, 1.0e-2}

  defp close?(a, b, atol, rtol), do: abs(a - b) <= atol + rtol * abs(b)

  defp rebuild(list, shape) do
    Nx.tensor(list, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.reshape(shape)
  end
end
