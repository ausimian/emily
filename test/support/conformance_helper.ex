defmodule Emily.ConformanceHelper do
  @moduledoc """
  Shared scaffolding for `test/emily/conformance/*` suites.

  `use Emily.ConformanceHelper` installs:

    * a per-test `setup` block that swaps the process-scoped default
      backend to `Emily.Backend` for the duration of the test and
      restores it on exit — pdict scope (not application env) so
      modules can run `async: true`;
    * an import of `assert_all_close/2,3`, the tolerance-aware
      comparison we use against reference slices produced by
      HuggingFace Transformers (PyTorch). Mirrors
      `Bumblebee.TestHelpers.assert_all_close` without pulling the
      whole Bumblebee test helper module in.

  Each conformance module still declares its own `@moduletag`s
  (`:conformance`, `:qwen3_full`, `:vit_full`, …) — those are not
  shared because they gate test selection.

  ## When to avoid this helper

  Tests that drive `Nx.Serving.batched_run` through a supervised
  serving process cannot rely on the pdict default, because the
  serving worker is a separate process that falls back to the
  application env. Those tests must set `Nx.global_default_backend`
  directly (and run `async: false`).
  """

  defmacro __using__(_opts) do
    quote do
      import Emily.ConformanceHelper, only: [assert_all_close: 2, assert_all_close: 3]

      setup do
        Nx.default_backend(Emily.Backend)
        :ok
      end
    end
  end

  @doc """
  Assert that every element of `left` agrees with `right` within
  `atol + rtol * |right|`.

  Materialises both tensors on `Nx.BinaryBackend` on failure so the
  diff in the ExUnit output is readable (an inspect on an
  `Emily.Backend` tensor would recurse through MLX).
  """
  def assert_all_close(left, right, opts \\ []) do
    atol = opts[:atol] || 1.0e-4
    rtol = opts[:rtol] || 1.0e-4

    equal_tensor =
      left
      |> Nx.all_close(right, atol: atol, rtol: rtol)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    if Nx.to_number(equal_tensor) != 1 do
      ExUnit.Assertions.flunk("""
      expected

      #{inspect(Nx.backend_copy(left, Nx.BinaryBackend))}

      to be within tolerance of

      #{inspect(Nx.backend_copy(right, Nx.BinaryBackend))}
      """)
    end
  end
end
