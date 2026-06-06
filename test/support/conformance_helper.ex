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
      import Emily.ConformanceHelper,
        only: [assert_all_close: 2, assert_all_close: 3, mode_test: 2, mode_test: 3]

      setup do
        Nx.default_backend(Emily.Backend)
        :ok
      end
    end
  end

  @doc """
  Define a conformance test in three lanes from a single body.

  Expands to three `ExUnit` tests that share `body` but bind a different
  `predict_opts` keyword list:

    * the default lane binds `predict_opts` to `[]` — the evaluator path
      Bumblebee/Axon use out of the box (the existing "eval'd" mode);
    * the native lane binds `predict_opts` to
      `[compiler: Emily.Compiler, native: true, native_fallback: :raise]`
      and is additionally tagged `:native`;
    * the fusion lane adds `fuse: true` (wrapping the replay in
      `mx::compile`) and is additionally tagged `:fuse`.

  The module is already tagged `:conformance`, so the native and fusion
  lanes carry that tag too: `mix test --only conformance` runs all three,
  while `mix test --only native` / `mix test --only fuse` run
  one lane each. Because every lane resolves the same HuggingFace repos,
  whichever runs first reads from `~/.cache/bumblebee` for the rest — the
  download is paid once.

  `mx::compile` reassociates f32, so the fusion lane's logits are not
  bit-identical to the evaluator's; it shares the same reference and
  tolerance as the other lanes (these tiny-random forwards drift well
  within `assert_all_close`'s default), and `assert_finite!`-style smoke
  tests are robust to the drift outright.

  The body must thread `predict_opts` into whatever drives the forward
  pass so the two lanes assert against the *identical* reference and
  cannot drift apart in maintenance:

      mode_test ":base" do
        {:ok, %{model: model, params: params}} = Bumblebee.load_model(...)
        outputs = Axon.predict(model, params, inputs, predict_opts)
        assert_all_close(outputs.hidden_state, ...)
      end

  For `Axon.build`-driven tests, build `init_fn` on the evaluator (params
  are random-init, mode-irrelevant) and only `predict_fn` under
  `predict_opts`, so the native lane gates the forward pass alone:

      {init_fn, _} = Axon.build(model)
      {_, predict_fn} = Axon.build(model, predict_opts)

  `native_fallback: :raise` makes the native lane a no-fallback gate: an
  op that does not lower fails the test rather than silently degrading to
  the evaluator, so a red native lane is a concrete op-coverage gap.

  ## Options

    * `:lane_tags` (default `true`) — when `false`, the native and fusion
      lanes are emitted *without* the cross-cutting `:native` /
      `:fuse` tags. The heavyweight `*_full` suites pass
      `lane_tags: false` so their compiler lanes stay gated behind the
      suite's own `:*_full` moduletag; otherwise `--only native` would
      start pulling full-size checkpoints. `--only vit_full` then runs all
      three lanes of that suite.

    * `:tag` — an extra tag stamped on *every* lane. Used by the
      `Nx.Serving` test, which lives in a `:conformance`-tagged module but
      must stay gated behind `:distilbert_full` like its eval lane:
      `tag: :distilbert_full, lane_tags: false`.
  """
  defmacro mode_test(name, opts \\ [], do: body) do
    tag_lanes? = Keyword.get(opts, :lane_tags, true)
    extra_tag = Keyword.get(opts, :tag)

    lanes = [
      lane([extra_tag], name, "", [], body),
      lane(
        [extra_tag, tag_lanes? && :native],
        name,
        " [native]",
        [compiler: Emily.Compiler, native: true, native_fallback: :raise],
        body
      ),
      lane(
        [extra_tag, tag_lanes? && :fuse],
        name,
        " [fuse]",
        [compiler: Emily.Compiler, native: true, native_fallback: :raise, fuse: true],
        body
      )
    ]

    quote do
      (unquote_splicing(lanes))
    end
  end

  # Build one `mode_test` lane: a `test` that binds `predict_opts` for the
  # body, preceded by one `@tag` per entry in `tags` (nil/false entries are
  # dropped). The `*_full` suites pass `lane_tags: false` to drop the
  # `:native` / `:fuse` tags and rely on their own `:*_full`
  # moduletag (or an explicit `:tag`) instead.
  defp lane(tags, name, suffix, predict_opts, body) do
    tags = Enum.reject(tags, &(&1 in [nil, false]))

    name_ast =
      if suffix == "", do: name, else: quote(do: unquote(name) <> unquote(suffix))

    tag_attrs = for t <- tags, do: quote(do: @tag(unquote(t)))

    quote do
      (unquote_splicing(tag_attrs))

      test unquote(name_ast) do
        var!(predict_opts) = unquote(predict_opts)
        unquote(body)
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
