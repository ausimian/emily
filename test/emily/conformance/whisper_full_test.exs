defmodule Emily.Conformance.WhisperFullTest do
  @moduledoc """
  Full `openai/whisper-tiny` end-to-end conformance test.

  Like the other `*_full` suites, this is excluded even from
  `mix test --only conformance`: the model is ~150 MB on first
  fetch. Run explicitly:

      mix test --only whisper_full

  Uses a deterministic synthetic mel-features tensor rather than a
  checked-in audio fixture, by the same reasoning as
  `vit_full_test.exs`: binary test assets in git are annoying, and
  the intent is to catch numerical drift on real-size weight
  tensors (encoder conv frontend + 4 encoder blocks + 4 decoder
  blocks with cross-attention), not to verify mel-spectrogram
  computation. Input shape is Whisper's canonical 30-second window:
  3000 time-steps × 80 mel bins.

  A failure means the backend has drifted, Bumblebee's Whisper port
  has changed, or the HF checkpoint has been republished — all of
  which are real signals.
  """

  use ExUnit.Case, async: true
  use Emily.ConformanceHelper

  alias Emily.Bumblebee.FastKernels

  @moduletag :whisper_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  test "openai/whisper-tiny forward pass matches pinned logits slice" do
    {:ok, %{model: model, params: params, spec: spec}} =
      Bumblebee.load_model({:hf, "openai/whisper-tiny"})

    assert %Bumblebee.Audio.Whisper{architecture: :for_conditional_generation} = spec

    # Synthetic 30 s mel window. Nx.iota + sin produces a
    # fully-deterministic, feature-rich signal — more useful than
    # a constant because it exercises the attention pattern rather
    # than landing on a degenerate uniform hidden state.
    input_features =
      Nx.sin(Nx.iota({1, 3000, 80}, type: :f32) |> Nx.multiply(0.01))

    # Short decoder prompt: the four Whisper special tokens that open
    # every English-language transcription (<|startoftranscript|>,
    # <|en|>, <|transcribe|>, <|notimestamps|>) plus two text-token
    # placeholders. Exact ids don't matter for a numerical pin —
    # they just need to be in-vocab and deterministic.
    decoder_input_ids = Nx.tensor([[50_258, 50_259, 50_359, 50_363, 50, 100]])
    decoder_attention_mask = Nx.tensor([[1, 1, 1, 1, 1, 1]])

    inputs = %{
      "input_features" => input_features,
      "decoder_input_ids" => decoder_input_ids,
      "decoder_attention_mask" => decoder_attention_mask
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 6, 51_865}

    argmax =
      outputs.logits[[.., -1, ..]]
      |> Nx.argmax(axis: -1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> hd()

    # On the synthetic mel input the decoder collapses to
    # <|endoftext|> (50257) by the last position. That's fine for a
    # pin — the assertion is "the same backend + the same weights +
    # the same input reproduce the same token", not "the model says
    # something interesting".
    assert argmax == 50_257

    assert_all_close(
      outputs.logits[[.., 0..2, 0..2]],
      Nx.tensor([
        [[2.9246, 0.2663, 3.8530], [-4.5523, -8.4833, -4.4232], [17.7350, 16.3070, 13.2149]]
      ])
    )
  end

  @tag :fast_kernels_full
  test "Whisper-tiny with fused MLX kernels matches the pinned argmax within widened tolerance" do
    {:ok, %{model: model, params: params}} =
      Bumblebee.load_model({:hf, "openai/whisper-tiny"})

    fast_model = FastKernels.apply(model)

    input_features =
      Nx.sin(Nx.iota({1, 3000, 80}, type: :f32) |> Nx.multiply(0.01))

    decoder_input_ids = Nx.tensor([[50_258, 50_259, 50_359, 50_363, 50, 100]])
    decoder_attention_mask = Nx.tensor([[1, 1, 1, 1, 1, 1]])

    inputs = %{
      "input_features" => input_features,
      "decoder_input_ids" => decoder_input_ids,
      "decoder_attention_mask" => decoder_attention_mask
    }

    outputs = Axon.predict(fast_model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 6, 51_865}

    argmax =
      outputs.logits[[.., -1, ..]]
      |> Nx.argmax(axis: -1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> hd()

    assert argmax == 50_257

    # Fused LayerNorm + SDPA path: same logits-slice pin, with the
    # tolerance loosened one further OOM to absorb cross-attention's
    # extra fused-kernel reordering.
    assert_all_close(
      outputs.logits[[.., 0..2, 0..2]],
      Nx.tensor([
        [[2.9246, 0.2663, 3.8530], [-4.5523, -8.4833, -4.4232], [17.7350, 16.3070, 13.2149]]
      ]),
      atol: 1.0e-2,
      rtol: 1.0e-2
    )
  end
end
