defmodule Emily.TrainingHelper do
  @moduledoc """
  Handwritten training-loop scaffolding for M9 curve-matching tests
  (Phase D) and the training memory soak (Phase E).

  Deliberately no Axon — keeps the failure surface small so a red
  curve-match test points at grad/backend numerics, not at an
  indirect Axon layer. Axon enters the test surface only in Phase F
  (`:training_full` MNIST convergence canary).

  Two building blocks:

    * **MLP** — 2-layer ReLU MLP with explicit biases, MSE loss, vanilla
      SGD. Four parameter tensors (`w1`, `b1`, `w2`, `b2`).
    * **Transformer block** — single-head attention + 2-layer MLP +
      residuals + layer-norm-free. Parameters: `wq`, `wk`, `wv`, `wo`,
      `w_ff1`, `b_ff1`, `w_ff2`, `b_ff2`. Loss is mean-square of the
      block output against a fixed target.

  Both builders accept a `backend` option so parameters / inputs land
  on the matching backend. Shared by both backends for deterministic
  starting-point parity.
  """

  import Nx.Defn

  # -------------------- MLP --------------------

  @doc """
  Deterministic parameter init for a 2-layer MLP. Same seed + same
  dims + same backend produces bit-identical weights across calls, and
  transferring to another backend preserves that bit identity.

  `seed` just shifts the deterministic pattern; pick any integer.
  """
  def init_mlp({in_dim, hidden, out_dim}, seed, backend) do
    %{
      w1: det_weights({in_dim, hidden}, seed * 13 + 1, backend),
      b1: Nx.broadcast(0.0, {hidden}) |> Nx.backend_transfer(backend),
      w2: det_weights({hidden, out_dim}, seed * 13 + 2, backend),
      b2: Nx.broadcast(0.0, {out_dim}) |> Nx.backend_transfer(backend)
    }
  end

  @doc """
  Synthetic training batch: `{x, y}` with deterministic values on the
  given backend. `x` shape = `{batch, in_dim}`, `y` shape =
  `{batch, out_dim}`.
  """
  def mlp_batch({batch, in_dim, out_dim}, backend) do
    x = det_weights({batch, in_dim}, 999, backend)
    y = det_weights({batch, out_dim}, 1001, backend)
    {x, y}
  end

  defn mlp_loss(params, x, y) do
    z1 = Nx.dot(x, params.w1) + params.b1
    a1 = Nx.max(z1, 0.0)
    z2 = Nx.dot(a1, params.w2) + params.b2
    diff = z2 - y
    Nx.mean(diff * diff)
  end

  defn mlp_step_with_loss(params, x, y, lr) do
    loss = mlp_loss(params, x, y)
    grads = grad(params, fn p -> mlp_loss(p, x, y) end)

    new_params = %{
      w1: params.w1 - lr * grads.w1,
      b1: params.b1 - lr * grads.b1,
      w2: params.w2 - lr * grads.w2,
      b2: params.b2 - lr * grads.b2
    }

    {new_params, loss}
  end

  # -------------------- Mixed-precision MLP --------------------

  defn mlp_mp_step_with_loss(params, x, y, lr, loss_scale) do
    loss = mlp_mp_loss(params, x, y)
    grads = grad(params, fn p -> mlp_mp_loss(p, x, y) * loss_scale end)

    inv_scale = 1.0 / loss_scale

    new_params = %{
      w1: params.w1 - lr * (grads.w1 * inv_scale),
      b1: params.b1 - lr * (grads.b1 * inv_scale),
      w2: params.w2 - lr * (grads.w2 * inv_scale),
      b2: params.b2 - lr * (grads.b2 * inv_scale)
    }

    {new_params, loss}
  end

  defnp mlp_mp_loss(params, x, y) do
    w1 = Nx.as_type(params.w1, {:bf, 16})
    b1 = Nx.as_type(params.b1, {:bf, 16})
    w2 = Nx.as_type(params.w2, {:bf, 16})
    b2 = Nx.as_type(params.b2, {:bf, 16})

    z1 = Nx.dot(x, w1) + b1
    a1 = Nx.max(z1, 0.0)
    z2 = Nx.dot(a1, w2) + b2
    diff = z2 - y
    Nx.mean(diff * diff)
  end

  # -------------------- Small CNN (M17 curve test) --------------------

  @doc """
  Deterministic parameter init for a tiny LeNet-style CNN:
  conv(1->4, 3x3) → maxpool(2x2) → conv(4->8, 3x3) → maxpool(2x2) →
  dense(8*out_h*out_w -> classes).

  `input_shape` = `{channels=1, h, w}`. `classes` is the output logit
  count. Returns a params map including the computed flatten size so
  the forward pass stays defn-compatible without introspection.
  """
  def init_cnn({_ch_in, h, w}, classes, seed, backend) do
    # After conv(3x3,valid) + maxpool(2,stride 2) twice:
    # h1 = h - 2; h_after_pool1 = div(h1, 2)
    # h2 = h_after_pool1 - 2; h_after_pool2 = div(h2, 2)
    h_out = div(div(h - 2, 2) - 2, 2)
    w_out = div(div(w - 2, 2) - 2, 2)
    flatten_size = 8 * h_out * w_out

    %{
      k1: det_weights({4, 1, 3, 3}, seed * 17 + 1, backend),
      b1: Nx.broadcast(0.0, {4}) |> Nx.backend_transfer(backend),
      k2: det_weights({8, 4, 3, 3}, seed * 17 + 2, backend),
      b2: Nx.broadcast(0.0, {8}) |> Nx.backend_transfer(backend),
      w_fc: det_weights({flatten_size, classes}, seed * 17 + 3, backend),
      b_fc: Nx.broadcast(0.0, {classes}) |> Nx.backend_transfer(backend)
    }
  end

  @doc """
  Synthetic CNN training batch. `x` shape = `{batch, 1, h, w}`,
  `y` shape = `{batch, classes}` (soft targets).
  """
  def cnn_batch({batch, h, w}, classes, backend) do
    x = det_weights({batch, 1, h, w}, 2001, backend)
    y = det_weights({batch, classes}, 2003, backend)
    {x, y}
  end

  defn cnn_forward(params, x) do
    # Block 1: conv → relu → maxpool.
    h1 = Nx.conv(x, params.k1) + Nx.reshape(params.b1, {1, 4, 1, 1})
    h1 = Nx.max(h1, 0.0)
    h1 = Nx.window_max(h1, {1, 1, 2, 2}, strides: [1, 1, 2, 2])

    # Block 2: conv → relu → maxpool.
    h2 = Nx.conv(h1, params.k2) + Nx.reshape(params.b2, {1, 8, 1, 1})
    h2 = Nx.max(h2, 0.0)
    h2 = Nx.window_max(h2, {1, 1, 2, 2}, strides: [1, 1, 2, 2])

    # Flatten + FC head.
    batch = Nx.axis_size(h2, 0)
    flat = Nx.reshape(h2, {batch, :auto})
    Nx.dot(flat, params.w_fc) + params.b_fc
  end

  defn cnn_loss(params, x, y) do
    logits = cnn_forward(params, x)
    diff = logits - y
    Nx.mean(diff * diff)
  end

  defn cnn_step_with_loss(params, x, y, lr) do
    loss = cnn_loss(params, x, y)
    grads = grad(params, fn p -> cnn_loss(p, x, y) end)

    new_params = %{
      k1: params.k1 - lr * grads.k1,
      b1: params.b1 - lr * grads.b1,
      k2: params.k2 - lr * grads.k2,
      b2: params.b2 - lr * grads.b2,
      w_fc: params.w_fc - lr * grads.w_fc,
      b_fc: params.b_fc - lr * grads.b_fc
    }

    {new_params, loss}
  end

  # -------------------- Transformer block --------------------

  @doc "Init the transformer-block parameters for `{embed_dim, ff_dim}`."
  def init_block({embed, ff}, seed, backend) do
    %{
      wq: det_weights({embed, embed}, seed * 31 + 1, backend),
      wk: det_weights({embed, embed}, seed * 31 + 2, backend),
      wv: det_weights({embed, embed}, seed * 31 + 3, backend),
      wo: det_weights({embed, embed}, seed * 31 + 4, backend),
      w_ff1: det_weights({embed, ff}, seed * 31 + 5, backend),
      b_ff1: Nx.broadcast(0.0, {ff}) |> Nx.backend_transfer(backend),
      w_ff2: det_weights({ff, embed}, seed * 31 + 6, backend),
      b_ff2: Nx.broadcast(0.0, {embed}) |> Nx.backend_transfer(backend)
    }
  end

  @doc """
  Block batch: `{x, y}` with shapes `{seq, embed}` each (no batch
  dim — single-sequence training for simplicity; extends trivially).
  """
  def block_batch({seq, embed}, backend) do
    x = det_weights({seq, embed}, 777, backend)
    y = det_weights({seq, embed}, 888, backend)
    {x, y}
  end

  defn block_forward(params, x, scale) do
    # Single-head self-attention with scaled dot-product + residual.
    # `scale` is `1 / sqrt(embed_dim)` passed in as a scalar tensor so
    # defn doesn't have to introspect shapes.
    q = Nx.dot(x, params.wq)
    k = Nx.dot(x, params.wk)
    v = Nx.dot(x, params.wv)
    logits = Nx.dot(q, Nx.transpose(k)) * scale
    attn = softmax_last(logits)
    attended = Nx.dot(attn, v) |> Nx.dot(params.wo)
    h = x + attended

    # FFN + residual.
    ff = Nx.max(Nx.dot(h, params.w_ff1) + params.b_ff1, 0.0)
    out = Nx.dot(ff, params.w_ff2) + params.b_ff2
    h + out
  end

  defn block_loss(params, x, y, scale) do
    out = block_forward(params, x, scale)
    diff = out - y
    Nx.mean(diff * diff)
  end

  defn block_step_with_loss(params, x, y, lr, scale) do
    loss = block_loss(params, x, y, scale)
    grads = grad(params, fn p -> block_loss(p, x, y, scale) end)

    new_params = %{
      wq: params.wq - lr * grads.wq,
      wk: params.wk - lr * grads.wk,
      wv: params.wv - lr * grads.wv,
      wo: params.wo - lr * grads.wo,
      w_ff1: params.w_ff1 - lr * grads.w_ff1,
      b_ff1: params.b_ff1 - lr * grads.b_ff1,
      w_ff2: params.w_ff2 - lr * grads.w_ff2,
      b_ff2: params.b_ff2 - lr * grads.b_ff2
    }

    {new_params, loss}
  end

  # -------------------- Driver --------------------

  @doc """
  Run `n` training steps under the given compiler, collecting the
  per-step loss (pre-update) as a list of Elixir floats.

  `step_fun` has signature `(params, args...) -> {new_params, loss}`.
  `args` is a list of the remaining non-params tensors passed after
  params on each call — e.g. `[x, y, lr]` for MLP, `[x, y, lr, scale]`
  for the transformer block.
  """
  def run_steps(step_fun, params, args, n, compiler) when is_list(args) do
    {_final, losses_rev} =
      Enum.reduce(1..n, {params, []}, fn _i, {params, losses} ->
        {new_params, loss} =
          Nx.Defn.jit_apply(step_fun, [params | args], compiler: compiler)

        loss_f = loss |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number()
        {new_params, [loss_f | losses]}
      end)

    Enum.reverse(losses_rev)
  end

  # -------------------- Curve-matching assertions --------------------

  @doc """
  Returns true if `a` and `b` are within `atol + rtol * abs(b)`.
  """
  def close?(a, b, atol, rtol), do: abs(a - b) <= atol + rtol * abs(b)

  @doc """
  Flunks with a trajectory diff when per-step losses diverge.
  """
  def flunk_trajectory(i, le, lb, losses_emily, losses_bin) do
    preview_e = losses_emily |> Enum.take(min(i + 3, length(losses_emily)))
    preview_b = losses_bin |> Enum.take(min(i + 3, length(losses_bin)))

    ExUnit.Assertions.flunk("""
    per-step loss diverged at step #{i}:
      emily=#{le} bin=#{lb} reldiff=#{abs(le - lb) / abs(lb)}
    emily trajectory (first #{length(preview_e)} steps): #{inspect(preview_e)}
    bin   trajectory (first #{length(preview_b)} steps): #{inspect(preview_b)}
    """)
  end

  # -------------------- Internal --------------------

  # Deterministic, backend-agnostic "random-ish" weight init. Build the
  # tensor on BinaryBackend for bit-identical starting points across
  # backends, then transfer. Values lie in (-0.3, 0.3) with a spatially
  # non-symmetric pattern so layer outputs aren't degenerate.
  defp det_weights(shape, seed, backend) do
    size = shape |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

    Nx.iota({size}, type: {:f, 32}, backend: Nx.BinaryBackend)
    |> Nx.multiply(0.7)
    |> Nx.add(seed * 7.1)
    |> Nx.sin()
    |> Nx.multiply(0.3)
    |> Nx.reshape(shape)
    |> Nx.backend_transfer(backend)
  end

  # Numerically stable softmax along the last axis — defn-compatible.
  defn softmax_last(t) do
    m = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    e = Nx.exp(t - m)
    e / Nx.sum(e, axes: [-1], keep_axes: true)
  end
end
