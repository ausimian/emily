defmodule Emily.Conformance.VitFullTest do
  @moduledoc """
  Full `google/vit-base-patch16-224` end-to-end conformance test.

  Like `Qwen3FullTest`, this is excluded even from
  `mix test --only conformance`: the model is ~330 MB on first fetch,
  so running it on every push is the wrong default. Run explicitly:

      mix test --only vit_full

  The reference slice pinned below is the forward-pass output of the
  full-size ViT weights on `Emily.Backend`, run against a
  deterministic synthetic pixel tensor. Using a synthetic input
  rather than a checked-in JPEG keeps binary fixtures out of the
  repo and avoids a featurizer round-trip — the goal is to catch
  numerical drift on real-size weight tensors, not to verify image
  preprocessing.

  A failure means the backend has drifted, Bumblebee's ViT port has
  changed, or the HF checkpoint has been republished — all of which
  are real signals.
  """

  use ExUnit.Case, async: false
  use Emily.ConformanceHelper

  alias Emily.Bumblebee.FastKernels

  @moduletag :vit_full
  @moduletag capture_log: true
  @moduletag timeout: 600_000

  test "google/vit-base-patch16-224 forward pass matches pinned logits slice" do
    {:ok, %{model: model, params: params, spec: spec}} =
      Bumblebee.load_model({:hf, "google/vit-base-patch16-224"})

    assert %Bumblebee.Vision.Vit{architecture: :for_image_classification} = spec
    # Full ViT-Base: 224×224×3 input, 1000 ImageNet classes.
    assert spec.num_labels == 1000

    inputs = %{
      "pixel_values" => Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 224, 224, 3})
    }

    outputs = Axon.predict(model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 1000}

    # Leading 5 logits on a constant-0.5 gray input. Pinned after
    # validating on Apple Silicon (Emily.Backend on GPU). Tolerance
    # is the standard 1e-4; f16/bf16 accumulation drift on a
    # 12-layer forward pass is well within that.
    # Argmax first — a flat gray image classifies deterministically
    # as some ImageNet class. Whichever class it is, pin the index so
    # any numerical shift that flips the argmax surfaces here.
    argmax =
      outputs.logits
      |> Nx.argmax(axis: -1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> hd()

    assert argmax == 763

    assert_all_close(
      outputs.logits[[.., 0..4]],
      Nx.tensor([[0.0112, -0.5066, -0.7792, -1.0436, -0.1899]])
    )
  end

  @tag :fast_kernels_full
  test "ViT with fused MLX kernels matches the pinned argmax within widened tolerance" do
    {:ok, %{model: model, params: params}} =
      Bumblebee.load_model({:hf, "google/vit-base-patch16-224"})

    fast_model = FastKernels.apply(model)

    inputs = %{
      "pixel_values" => Nx.broadcast(Nx.tensor(0.5, type: :f32), {1, 224, 224, 3})
    }

    outputs = Axon.predict(fast_model, params, inputs)

    assert Nx.shape(outputs.logits) == {1, 1000}

    argmax =
      outputs.logits
      |> Nx.argmax(axis: -1)
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> hd()

    assert argmax == 763

    # Fused LayerNorm + SDPA reorder ops slightly; loosen tolerance by
    # ~10× over the pinned-logits assertion above. Empirical gap on
    # M3 was ~3e-4 across 12 layers; 1e-3 is comfortable.
    assert_all_close(
      outputs.logits[[.., 0..4]],
      Nx.tensor([[0.0112, -0.5066, -0.7792, -1.0436, -0.1899]]),
      atol: 1.0e-3,
      rtol: 1.0e-3
    )
  end
end
