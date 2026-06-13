defmodule Emily.Conformance.ModernBertTest do
  @moduledoc """
  Smoke test for Bumblebee 0.7's `Bumblebee.Text.ModernBert` on
  `Emily.Backend`.

  ModernBERT is a long-context (8192-token) BERT variant with RoPE
  positional encoding, GeGLU activation, and alternating local /
  global attention. This test drives the encoder end-to-end on Emily
  through a small synthetic spec — no HuggingFace download required.

  Of the three Bumblebee 0.7 smoke tests this is the most interesting
  for Emily: it's the first encoder with rotary embeddings in the
  conformance suite, and it exercises both the local and global
  attention layers in alternating order.

  Verifies:

    * the `:base` Axon model builds,
    * `init_fn`/`predict_fn` run on `Emily.Backend` without raising,
    * the hidden-state output has the expected shape and contains no
      NaN/Inf.

  Tagged `:conformance`:

      mix test --only conformance
  """

  use ExUnit.Case, async: true
  use Emily.ConformanceHelper

  alias Bumblebee.Text.ModernBert

  @moduletag :conformance
  @moduletag capture_log: true

  mode_test "ModernBert :base forward on Emily.Backend" do
    spec =
      Bumblebee.configure(ModernBert,
        architecture: :base,
        vocab_size: 32,
        max_positions: 32,
        hidden_size: 16,
        num_blocks: 3,
        num_attention_heads: 2,
        intermediate_size: 32,
        local_attention_window: 4
      )

    model = ModernBert.model(spec)
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
      "input_ids" => Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 0]]),
      "attention_mask" => Nx.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
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
