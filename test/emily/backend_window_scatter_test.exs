defmodule Emily.BackendWindowScatterTest do
  @moduledoc """
  Oracle tests for `Emily.Backend.window_scatter_max/6` and
  `window_scatter_min/6` — the M17 lifts of Nx's select-and-scatter
  primitives that the symbolic `grad(window_max)` /
  `grad(window_min)` rewrite into.

  Covers:
    * 1-D / 2-D / 3-D inputs
    * `:valid` / `:same` / explicit padding
    * Overlapping windows that select the same output position
      (source contributions must sum via `scatter_add` semantics)
    * Tie-break — Nx picks the **last-occurrence** of the argmax/min
      within a window; MLX's native `argmax` picks first, so the
      C++ path uses a mask-times-position trick to recover
      last-occurrence. Regressions would show as misplaced values.
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 3]

  defp emily(tensor), do: Nx.backend_transfer(tensor, Emily.Backend)
  defp bin(tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  defp run_scatter(op, t, source, init, window_shape, opts) do
    emily_result =
      apply(Nx, op, [emily(t), emily(source), emily(init), window_shape, opts])

    ref_result =
      apply(Nx, op, [bin(t), bin(source), bin(init), window_shape, opts])

    assert_close(emily_result, ref_result, tol: 1.0e-5)
  end

  describe "window_scatter_max" do
    test "2-D, :valid padding, non-overlapping windows" do
      t =
        Nx.tensor(
          [
            [7, 2, 5, 3, 10, 2],
            [3, 8, 9, 3, 4, 2],
            [1, 5, 7, 5, 6, 1],
            [0, 6, 2, 7, 2, 8]
          ],
          type: {:f, 32}
        )

      source = Nx.tensor([[2, 6], [3, 1]], type: {:f, 32})
      init = Nx.tensor(0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {2, 3}, strides: [2, 3])
    end

    test "2-D, strides == kernel (non-overlapping pools)" do
      t =
        Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend)
        |> Nx.add(1.0)

      source = Nx.tensor([[10.0, 20.0], [30.0, 40.0]], type: {:f, 32})
      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {2, 2}, strides: [2, 2])
    end

    test "2-D, overlapping windows with stride 1" do
      # Overlapping windows can pick the same input position as argmax;
      # scatter_add accumulates each window's source contribution there.
      t =
        Nx.tensor(
          [
            [1.0, 5.0, 2.0, 1.0],
            [3.0, 2.0, 4.0, 6.0],
            [2.0, 1.0, 3.0, 2.0]
          ],
          type: {:f, 32}
        )

      source =
        Nx.tensor(
          [
            [100.0, 200.0, 300.0],
            [400.0, 500.0, 600.0]
          ],
          type: {:f, 32}
        )

      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {2, 2}, strides: [1, 1])
    end

    test ":same padding — selected position can land in padding region" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
      source = Nx.tensor([[10.0, 20.0], [30.0, 40.0]], type: {:f, 32})
      init = Nx.tensor(0.0, type: {:f, 32})

      run_scatter(:window_scatter_max, t, source, init, {2, 2},
        padding: :same,
        strides: [1, 1]
      )
    end

    test "tie-break — last-occurrence wins inside a window" do
      # The middle two columns both equal 3 within each {1, 2} window
      # at cols 1/2 and cols 2/3. Nx picks the LAST 3. Verifies the
      # mask*pos argmax trick in the C++ path matches Nx's semantics.
      t = Nx.tensor([[1.0, 3.0, 3.0, 1.0]], type: {:f, 32})
      source = Nx.tensor([[5.0, 7.0, 9.0]], type: {:f, 32})
      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {1, 2}, strides: [1, 1])
    end

    test "non-zero init value contributes to unselected positions" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: {:f, 32})
      source = Nx.tensor([[10.0]], type: {:f, 32})
      init = Nx.tensor(-1.0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {2, 2}, strides: [2, 2])
    end

    test "1-D input" do
      t = Nx.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], type: {:f, 32})
      source = Nx.tensor([10.0, 20.0, 30.0], type: {:f, 32})
      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {2}, strides: [2])
    end

    test "3-D input" do
      t =
        Nx.iota({2, 3, 4}, type: {:f, 32}, backend: Nx.BinaryBackend)
        |> Nx.add(1.0)

      source = Nx.iota({1, 2, 2}, type: {:f, 32}, backend: Nx.BinaryBackend)
      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_max, t, source, init, {2, 2, 2}, strides: [2, 1, 2])
    end
  end

  describe "window_scatter_min" do
    test "2-D from Nx docstring" do
      t =
        Nx.tensor(
          [
            [7, 2, 5, 3, 10, 2],
            [3, 8, 9, 3, 4, 2],
            [1, 5, 7, 5, 6, 1],
            [0, 6, 2, 7, 2, 8]
          ],
          type: {:f, 32}
        )

      source = Nx.tensor([[2, 6], [3, 1]], type: {:f, 32})
      init = Nx.tensor(0, type: {:f, 32})
      run_scatter(:window_scatter_min, t, source, init, {2, 3}, strides: [2, 3])
    end

    test "overlapping windows" do
      t =
        Nx.tensor(
          [
            [5.0, 1.0, 4.0],
            [2.0, 3.0, 6.0]
          ],
          type: {:f, 32}
        )

      source = Nx.tensor([[10.0, 20.0], [30.0, 40.0]], type: {:f, 32})
      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_min, t, source, init, {1, 2}, strides: [1, 1])
    end

    test "tie-break — last-occurrence wins (min)" do
      t = Nx.tensor([[5.0, 2.0, 2.0, 5.0]], type: {:f, 32})
      source = Nx.tensor([[7.0, 9.0, 11.0]], type: {:f, 32})
      init = Nx.tensor(0.0, type: {:f, 32})
      run_scatter(:window_scatter_min, t, source, init, {1, 2}, strides: [1, 1])
    end
  end
end
