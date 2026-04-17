defmodule Emily.MixedPrecision do
  @moduledoc """
  Mixed-precision training utilities.

  Standard recipe for memory-efficient training: bf16 activations with
  f32 master weights and dynamic loss scaling. Keeps the full-precision
  copy of parameters for numerically stable optimizer updates while
  running the forward and backward pass in half precision.

  ## Worked example

      alias Emily.MixedPrecision, as: MP
      alias Emily.MixedPrecision.LossScaler

      # f32 master weights — the optimizer's ground truth.
      master_params = init_params()
      scaler = MP.loss_scale()

      for {x, y} <- batches, reduce: {master_params, scaler} do
        {params, scaler} ->
          # Forward pass in bf16.
          bf16_params = MP.cast_params(params, {:bf, 16})

          # Backward pass: grad w.r.t. f32 master params, but the
          # forward graph runs in bf16 thanks to the as_type casts
          # inside the closure.
          grads =
            Nx.Defn.grad(params, fn p ->
              p
              |> MP.cast_params({:bf, 16})
              |> forward(x, y)
              |> MP.scale_loss(scaler)
            end)

          # Unscale, detect overflow, adjust scaler.
          {grads, overflow?} = MP.unscale(grads, scaler)
          scaler = MP.update(scaler, overflow?)

          if overflow? do
            {params, scaler}
          else
            f32_grads = MP.accumulate_grad(grads, {:f, 32})
            {sgd_step(params, f32_grads, lr), scaler}
          end
      end

  ## Container traversal

  `cast_params/2`, `accumulate_grad/2`, and `has_overflow?/1` traverse
  plain maps, tuples, and lists of `Nx.Tensor` leaves. For
  `Axon.ModelState`, access the `.data` field first:

      MP.cast_params(model_state.data, {:bf, 16})
  """

  defmodule LossScaler do
    @moduledoc """
    Dynamic loss-scaler state for mixed-precision training.

    Tracks the current scale factor and the number of consecutive
    successful (non-overflow) steps. On overflow the scale is halved;
    after `growth_interval` successful steps it doubles.
    """

    @default_scale 65_536.0

    @enforce_keys [:scale]
    defstruct scale: @default_scale,
              growth_factor: 2.0,
              backoff_factor: 0.5,
              growth_interval: 2000,
              min_scale: 1.0,
              counter: 0

    @doc "Create a new loss scaler."
    def new(opts \\ []) do
      fields =
        opts
        |> Keyword.validate!([
          :scale,
          :growth_factor,
          :backoff_factor,
          :growth_interval,
          :min_scale
        ])
        |> then(&Keyword.merge(Map.to_list(%__MODULE__{scale: @default_scale}), &1))

      struct!(__MODULE__, fields)
    end
  end

  @doc """
  Downcast float tensors in a nested structure to `type`.

  Integer and predicate tensors are left unchanged.
  """
  def cast_params(params, type), do: deep_apply(params, &Nx.as_type(&1, type))

  @doc """
  Upcast float tensors in a nested gradient structure to `type`.

  Semantically identical to `cast_params/2` — exists for readability
  at the call site (the direction of the cast is part of the name).
  """
  def accumulate_grad(grads, type), do: deep_apply(grads, &Nx.as_type(&1, type))

  @doc """
  Create a new dynamic loss scaler.

  ## Options

    * `:scale` — initial scale factor (default `65_536.0`)
    * `:growth_factor` — multiply scale by this on growth (default `2.0`)
    * `:backoff_factor` — multiply scale by this on overflow (default `0.5`)
    * `:growth_interval` — successful steps before growing (default `2000`)
    * `:min_scale` — floor for the scale (default `1.0`)
  """
  def loss_scale(opts \\ []), do: LossScaler.new(opts)

  @doc """
  Scale the loss by the scaler's current factor.

  Call this inside the function passed to `Nx.Defn.grad` so that the
  backward pass produces scaled gradients.
  """
  def scale_loss(loss, %LossScaler{scale: scale}) do
    Nx.multiply(loss, scale)
  end

  @doc """
  Unscale gradients and detect overflow.

  Divides every float tensor in `grads` by `scaler.scale`, then checks
  for inf/nan. Returns `{unscaled_grads, overflow?}`.
  """
  def unscale(grads, %LossScaler{scale: scale}) do
    inv_scale = 1.0 / scale
    unscaled = deep_apply(grads, &Nx.multiply(&1, inv_scale))
    {unscaled, has_overflow?(unscaled)}
  end

  @doc """
  Update the scaler after a training step.

  On overflow: halves the scale (floored at `min_scale`), resets the
  counter. On success: increments the counter; doubles the scale after
  `growth_interval` consecutive successes.
  """
  def update(%LossScaler{} = scaler, true = _overflow) do
    %{scaler | scale: max(scaler.min_scale, scaler.scale * scaler.backoff_factor), counter: 0}
  end

  def update(%LossScaler{} = scaler, false = _overflow) do
    counter = scaler.counter + 1

    if counter >= scaler.growth_interval do
      %{scaler | scale: scaler.scale * scaler.growth_factor, counter: 0}
    else
      %{scaler | counter: counter}
    end
  end

  @doc """
  Check whether any tensor in a nested structure contains inf or nan.
  """
  def has_overflow?(structure), do: deep_overflow?(structure)

  defp float_type?({kind, _}) when kind in [:f, :bf], do: true
  defp float_type?(_), do: false

  defp deep_apply(%Nx.Tensor{} = t, fun) do
    if float_type?(t.type), do: fun.(t), else: t
  end

  defp deep_apply(map, fun) when is_map(map) and not is_struct(map) do
    Map.new(map, fn {k, v} -> {k, deep_apply(v, fun)} end)
  end

  defp deep_apply(tuple, fun) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&deep_apply(&1, fun)) |> List.to_tuple()
  end

  defp deep_apply(list, fun) when is_list(list) do
    Enum.map(list, &deep_apply(&1, fun))
  end

  defp deep_apply(other, _fun), do: other

  defp deep_overflow?(%Nx.Tensor{} = t) do
    if float_type?(t.type) do
      t
      |> Nx.is_nan()
      |> Nx.logical_or(Nx.is_infinity(t))
      |> Nx.any()
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_number() == 1
    else
      false
    end
  end

  defp deep_overflow?(map) when is_map(map) and not is_struct(map) do
    Enum.any?(map, fn {_k, v} -> deep_overflow?(v) end)
  end

  defp deep_overflow?(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.any?(&deep_overflow?/1)
  end

  defp deep_overflow?(list) when is_list(list) do
    Enum.any?(list, &deep_overflow?/1)
  end

  defp deep_overflow?(_), do: false
end
