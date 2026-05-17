defmodule Emily.Conformance.SmolLm3Test do
  @moduledoc """
  Smoke test for Bumblebee 0.7's `Bumblebee.Text.SmolLm3` on
  `Emily.Backend`.

  SmolLM3 is the small (≈3B params) Llama-style decoder ShipLab
  released alongside Llama-3. This test drives the decoder graph
  end-to-end on Emily through a small synthetic spec — no
  HuggingFace download required. It exercises the GQA + RoPE +
  RMSNorm paths Emily already supports for Qwen3, but for a model
  that ships with Bumblebee 0.7 for the first time.

  Verifies:

    * the `:for_causal_language_modeling` Axon model builds,
    * `init_fn`/`predict_fn` run on `Emily.Backend` without raising,
    * the logits output has the expected shape and contains no
      NaN/Inf.

  Tagged `:conformance`:

      mix test --only conformance
  """

  use ExUnit.Case, async: true
  use Emily.ConformanceHelper

  alias Bumblebee.Text.SmolLm3

  @moduletag :conformance
  @moduletag capture_log: true

  test "SmolLm3 :for_causal_language_modeling forward on Emily.Backend" do
    spec =
      Bumblebee.configure(SmolLm3,
        architecture: :for_causal_language_modeling,
        vocab_size: 32,
        max_positions: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_blocks: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2
      )

    model = SmolLm3.model(spec)
    {init_fn, predict_fn} = Axon.build(model)

    input_template = %{"input_ids" => Nx.template({1, 8}, :s64)}
    params = init_fn.(input_template, Axon.ModelState.empty())

    inputs = %{"input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
    outputs = predict_fn.(params, inputs)

    assert Nx.shape(outputs.logits) == {1, 8, 32}
    assert_finite!(outputs.logits)
  end

  defp assert_finite!(tensor) do
    any_nan = tensor |> Nx.is_nan() |> Nx.any() |> Nx.to_number()
    any_inf = tensor |> Nx.is_infinity() |> Nx.any() |> Nx.to_number()
    assert any_nan == 0, "forward pass produced NaN"
    assert any_inf == 0, "forward pass produced Inf"
  end
end
