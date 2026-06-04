defmodule Emily.Conformance.CompilerNativeTest do
  @moduledoc """
  CM5 — the **no-fallback** gate: real Bumblebee model forwards compile
  through the native single-NIF `Emily.Compiler` (`native: true`) and
  match the Evaluator-on-`Emily.Backend` path, with **zero** fallback to
  `Nx.BinaryBackend`.

  The native compiler lowers the whole `Nx.Defn.Expr` to one program and
  **raises** on any op it can't lower (no silent fallback by design), so a
  forward that completes proves full native op coverage for that model.
  We additionally fail on any `[:emily, :fallback, *]` telemetry — the
  Backend-level BinaryBackend fallback — so an op that silently round-trips
  to the host is caught too.

  Gated `:conformance` (downloads ~3 MB tiny-random HF fixtures); run with
  `mix test --only conformance`.
  """
  use ExUnit.Case, async: false

  @moduletag :conformance
  @moduletag timeout: 600_000

  setup do
    prev = Nx.default_backend()
    Nx.global_default_backend(Emily.Backend)
    on_exit(fn -> Nx.global_default_backend(prev) end)
    :ok
  end

  # Run `model`'s forward both ways and assert every output leaf agrees,
  # while asserting no Backend fallback fires on the native path.
  defp assert_native_matches(model, params, inputs) do
    {_init, native_predict} = Axon.build(model, compiler: Emily.Compiler, native: true)
    {_init, eval_predict} = Axon.build(model, compiler: Emily.Compiler)

    {native, fallbacks} = with_fallback_count(fn -> native_predict.(params, inputs) end)
    eval = eval_predict.(params, inputs)

    assert fallbacks == 0,
           "native compile path triggered #{fallbacks} Backend BinaryBackend fallback(s)"

    compare_outputs(native, eval)
  end

  # Count [:emily, :fallback, *] telemetry events during `fun`.
  defp with_fallback_count(fun) do
    ref = make_ref()
    me = self()
    id = {__MODULE__, ref}

    :telemetry.attach_many(
      id,
      [[:emily, :fallback, :start], [:emily, :fallback, :stop], [:emily, :fallback, :exception]],
      &__MODULE__.handle_fallback/4,
      {me, ref}
    )

    result = fun.()
    :telemetry.detach(id)

    count = drain(ref, 0)
    {result, count}
  end

  @doc false
  def handle_fallback(_event, _measure, _meta, {pid, ref}), do: send(pid, {ref, :fallback})

  defp drain(ref, n) do
    receive do
      {^ref, :fallback} -> drain(ref, n + 1)
    after
      0 -> n
    end
  end

  defp compare_outputs(%Nx.Tensor{} = native, %Nx.Tensor{} = eval) do
    assert native.shape == eval.shape
    assert native.type == eval.type
    # Same MLX kernels in the same order => exact; allow a tiny tolerance
    # against fp reassociation in the lazy-graph eval.
    n = Nx.to_flat_list(native)
    e = Nx.to_flat_list(eval)

    assert Enum.zip(n, e) |> Enum.all?(fn {a, b} -> abs(a - b) <= 1.0e-4 + 1.0e-4 * abs(b) end),
           "native vs evaluator outputs diverge beyond tolerance"
  end

  defp compare_outputs(native, eval) when is_map(native) and not is_struct(native) do
    for {k, nv} <- native, Map.has_key?(eval, k) do
      compare_outputs(nv, Map.fetch!(eval, k))
    end
  end

  # Axon.None placeholders, tuples, and other non-tensor leaves: skip.
  defp compare_outputs(_native, _eval), do: :ok

  test "tiny DistilBERT base forward: single-NIF native == evaluator, no fallback" do
    {:ok, %{model: model, params: params, spec: _spec}} =
      Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DistilBertModel"})

    inputs = %{
      "input_ids" => Nx.tensor([[1, 5, 7, 2, 3, 9]], backend: Emily.Backend),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1]], backend: Emily.Backend)
    }

    assert_native_matches(model, params, inputs)
  end

  test "tiny DistilBERT for-masked-LM forward: native == evaluator, no fallback" do
    {:ok, %{model: model, params: params}} =
      Bumblebee.load_model({:hf, "hf-internal-testing/tiny-random-DistilBertForMaskedLM"})

    inputs = %{
      "input_ids" => Nx.tensor([[1, 5, 7, 2]], backend: Emily.Backend),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1]], backend: Emily.Backend)
    }

    assert_native_matches(model, params, inputs)
  end
end
