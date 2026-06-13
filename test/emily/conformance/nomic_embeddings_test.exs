defmodule Emily.Conformance.NomicEmbeddingsTest do
  @moduledoc """
  Smoke test for Bumblebee 0.7's `Bumblebee.Text.NomicBert` on
  `Emily.Backend`.

  NomicBERT is a long-context (8192-token) encoder used as an
  embedding model (`nomic-embed-text-v1`). This test drives the
  encoder graph end-to-end on Emily through a small synthetic spec —
  no HuggingFace download required. It verifies that:

    * the Axon model builds from `Bumblebee.Text.NomicBert.model/1`,
    * `init_fn`/`predict_fn` run on `Emily.Backend` without raising,
    * the forward pass produces a finite-valued hidden state of the
      expected shape.

  Tagged `:conformance` so it runs with the rest of the conformance
  umbrella but stays out of the default `mix test` set:

      mix test --only conformance
  """

  use ExUnit.Case, async: true
  use Emily.ConformanceHelper

  alias Bumblebee.Text.NomicBert

  @moduletag :conformance
  @moduletag capture_log: true

  mode_test "NomicBert :base forward runs end-to-end on Emily.Backend" do
    spec =
      Bumblebee.configure(NomicBert,
        architecture: :base,
        vocab_size: 32,
        max_positions: 32,
        hidden_size: 16,
        num_blocks: 2,
        num_attention_heads: 2,
        intermediate_size: 32,
        max_token_types: 2
      )

    model = NomicBert.model(spec)
    # Init on the evaluator (params are random-init, mode-irrelevant);
    # gate only the forward pass under `predict_opts`.
    {init_fn, _} = Axon.build(model)
    {_, predict_fn} = Axon.build(model, predict_opts)

    input_template = %{
      "input_ids" => Nx.template({1, 8}, :s64),
      "attention_mask" => Nx.template({1, 8}, :s64)
    }

    params = init_fn.(input_template, Axon.ModelState.empty())

    inputs = %{
      "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 0, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    }

    outputs = predict_fn.(params, inputs)

    assert Nx.shape(outputs.hidden_state) == {1, 8, 16}
    assert_finite!(outputs.hidden_state)
  end

  defp assert_finite!(tensor) do
    any_nan = tensor |> Nx.is_nan() |> Nx.any() |> Nx.to_number()
    any_inf = tensor |> Nx.is_infinity() |> Nx.any() |> Nx.to_number()
    assert any_nan == 0, "forward pass produced NaN"
    assert any_inf == 0, "forward pass produced Inf"
  end
end
