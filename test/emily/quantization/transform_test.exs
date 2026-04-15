defmodule Emily.Quantization.TransformTest do
  @moduledoc """
  Tests for `Emily.Quantization.Transform` — the graph-rewrite plus
  model-state-swap plumbing that turns a stock Axon/Bumblebee model
  into one whose dense layers run through `Emily.Quantization.Layers.quantized_dense/4`.

  The round-trip test constructs a small Axon MLP (two dense layers),
  initializes it to get a float model state, then quantizes both the
  graph and the state. Predictions from the quantized pair must match
  the float-model predictions within int4 quantization tolerance.

  This is the same test shape `Axon.Quantization.quantize/2` uses in
  upstream Axon's quantization tests, adapted to Emily's MLX-affine
  scheme.
  """

  use ExUnit.Case, async: false

  alias Emily.Quantization.Transform
  alias Emily.QuantizedWeight

  setup_all do
    prev = Nx.default_backend()
    Nx.global_default_backend(Emily.Backend)
    on_exit(fn -> Nx.global_default_backend(prev) end)
    :ok
  end

  describe "quantize_dense_layers/2" do
    test "replaces :dense nodes with :quantized_dense" do
      model =
        Axon.input("features", shape: {nil, 128})
        |> Axon.dense(256, name: "fc1")
        |> Axon.relu()
        |> Axon.dense(64, name: "fc2")

      qmodel = Transform.quantize_dense_layers(model)
      props = Axon.properties(qmodel)

      assert props["fc1"] == :quantized_dense
      assert props["fc2"] == :quantized_dense
    end

    test "preserves non-:dense layers untouched" do
      model =
        Axon.input("features", shape: {nil, 128})
        |> Axon.dense(256, name: "fc1")
        |> Axon.relu(name: "act")

      qmodel = Transform.quantize_dense_layers(model)
      props = Axon.properties(qmodel)

      assert props["fc1"] == :quantized_dense
      assert props["act"] == :relu
    end
  end

  describe "quantize_model_state/3" do
    test "replaces :dense kernels with %QuantizedWeight{}, leaves biases float" do
      model =
        Axon.input("features", shape: {nil, 128})
        |> Axon.dense(256, name: "fc1")

      {init_fn, _predict_fn} = Axon.build(model, compiler: Nx.Defn.Evaluator)
      state = init_fn.(Nx.template({1, 128}, :f32), Axon.ModelState.empty())

      qstate = Transform.quantize_model_state(model, state, bits: 4, group_size: 64)

      assert %QuantizedWeight{} = qstate.data["fc1"]["kernel"]
      assert %Nx.Tensor{} = qstate.data["fc1"]["bias"]
      assert qstate.data["fc1"]["kernel"].bits == 4
      assert qstate.data["fc1"]["kernel"].group_size == 64
      assert qstate.data["fc1"]["kernel"].transpose == true
    end

    test "transpose=false leaves kernel layout unchanged (AWQ-style)" do
      model =
        Axon.input("features", shape: {nil, 2})
        |> Axon.dense(128, name: "fc1")

      {init_fn, _predict_fn} = Axon.build(model, compiler: Nx.Defn.Evaluator)
      state = init_fn.(Nx.template({1, 2}, :f32), Axon.ModelState.empty())

      qstate =
        Transform.quantize_model_state(model, state,
          bits: 4,
          group_size: 64,
          transpose: false
        )

      qw = qstate.data["fc1"]["kernel"]
      assert qw.transpose == false
      # With transpose=false, stored value shape packs the last axis of
      # the original kernel [in, out] = [2, 128]. Packed last = 128 / 8 = 16.
      assert Nx.shape(qw.value) == {2, 16}
    end
  end

  describe "quantize/3 — round-trip" do
    # Per-element int4 error accumulates across layers and activations;
    # dense-model predictions are typically O(1) after glorot_uniform
    # init, so a 15% relative tolerance catches drift without flaking.
    @rel_tol 0.15

    test "quantized 2-layer MLP predicts close to dense (Nx.Defn.Evaluator)" do
      model =
        Axon.input("features", shape: {nil, 128})
        |> Axon.dense(256, name: "fc1")
        |> Axon.relu()
        |> Axon.dense(64, name: "fc2")

      assert_roundtrip_close(model, {3, 128}, compiler: Nx.Defn.Evaluator)
    end

    # Emily.Compiler is the path Bumblebee servings take when
    # configured with `defn_options: [compiler: Emily.Compiler]`.
    test "quantized single-dense predicts close to dense (Emily.Compiler)" do
      model =
        Axon.input("features", shape: {nil, 64})
        |> Axon.dense(128, name: "fc1")

      assert_roundtrip_close(model, {2, 64}, compiler: Emily.Compiler)
    end

    defp assert_roundtrip_close(model, input_shape, compiler: compiler) do
      {init_fn, predict_fn} = Axon.build(model, compiler: compiler)

      x =
        input_shape
        |> Nx.iota(backend: Emily.Backend, type: :f32)
        |> Nx.divide(elem(input_shape, tuple_size(input_shape) - 1))
        |> Nx.subtract(0.5)

      state = init_fn.(x, Axon.ModelState.empty())
      expected = predict_fn.(state, x)

      {qmodel, qstate} =
        Transform.quantize(model, state, bits: 4, group_size: 64, transpose: true)

      {_qinit_fn, qpredict_fn} = Axon.build(qmodel, compiler: compiler)
      actual = qpredict_fn.(qstate, x)

      assert Nx.shape(actual) == Nx.shape(expected)

      expected_list = Nx.to_flat_list(expected)
      actual_list = Nx.to_flat_list(actual)

      max_abs_err =
        expected_list
        |> Enum.zip(actual_list)
        |> Enum.map(fn {e, a} -> abs(e - a) end)
        |> Enum.max()

      max_ref = expected_list |> Enum.map(&abs/1) |> Enum.max()
      rel_err = max_abs_err / max(max_ref, 1.0e-6)

      assert rel_err < @rel_tol,
             "relative error #{rel_err} exceeded #{@rel_tol * 100}% tolerance"
    end
  end

  describe "validation" do
    test "raises on unsupported bits" do
      model = Axon.input("features", shape: {nil, 64}) |> Axon.dense(32)

      assert_raise ArgumentError, ~r/:bits must be one of/, fn ->
        Transform.quantize_dense_layers(model, bits: 5)
      end
    end

    test "raises on unknown option" do
      model = Axon.input("features", shape: {nil, 64}) |> Axon.dense(32)

      assert_raise ArgumentError, ~r/unknown keys \[:banana\]/, fn ->
        Transform.quantize_dense_layers(model, banana: true)
      end
    end
  end
end
