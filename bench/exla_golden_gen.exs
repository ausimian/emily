# Generates EXLA golden gradient values for Emily's M13 conformance suite.
#
# Usage:
#   elixir bench/exla_golden_gen.exs
#
# Writes test/support/exla_golden_data.ex directly (path relative to the
# script). On Linux+CUDA, set EXLA_TARGET=cuda before running.
#
# Regenerate when any of these change:
#   - A defn function in Emily.GradZoo (test/support/grad_zoo.ex)
#   - The fixed_inputs/1 builders in GradZoo
#   - The training step functions in Emily.TrainingHelper
#   - The EXLA or Nx version

Mix.install([{:nx, "~> 0.10"}, {:exla, "~> 0.10"}])

defmodule GoldenGen do
  import Nx.Defn

  # -------------------- Zoo functions --------------------
  # Verbatim copies from Emily.GradZoo.

  defn(grad_sum_op(x), do: grad(x, fn z -> Nx.sum(z) end))

  defn(grad_dot_left(x, b), do: grad(x, fn z -> z |> Nx.dot(b) |> Nx.sum() end))

  defn grad_reshape_transpose(x) do
    grad(x, fn z ->
      z |> Nx.transpose(axes: [1, 0]) |> Nx.reshape({12}) |> Nx.sum()
    end)
  end

  defn grad_broadcast(x) do
    grad(x, fn z -> z |> Nx.broadcast({4, 3}) |> Nx.sum() end)
  end

  defn grad_gather(x, idx) do
    grad(x, fn z -> z |> Nx.gather(idx, axes: [0, 1]) |> Nx.sum() end)
  end

  defn grad_indexed_add(x, idx, upd) do
    grad(x, fn z -> z |> Nx.indexed_add(idx, upd) |> Nx.sum() end)
  end

  defn grad_gather_dot_softmax(x, idx, w) do
    grad(x, fn z ->
      z
      |> Nx.gather(idx, axes: [0])
      |> Nx.reshape({3, 6})
      |> Nx.dot(w)
      |> softmax_last()
      |> Nx.sum()
    end)
  end

  defn grad_attention(x, wq, wk, wv, scale) do
    grad(x, fn z ->
      q = Nx.dot(z, wq)
      k = Nx.dot(z, wk)
      v = Nx.dot(z, wv)
      logits = Nx.dot(q, Nx.transpose(k)) * scale
      attn = softmax_last(logits)
      attn |> Nx.dot(v) |> Nx.sum()
    end)
  end

  defn softmax_last(t) do
    m = Nx.reduce_max(t, axes: [-1], keep_axes: true)
    e = Nx.exp(t - m)
    e / Nx.sum(e, axes: [-1], keep_axes: true)
  end

  # M17 window ops — kernel on spatial axes of an NCHW-shaped tensor.

  defn grad_window_sum(x) do
    grad(x, fn z -> z |> Nx.window_sum({1, 1, 2, 2}, strides: [1, 1, 1, 1]) |> Nx.sum() end)
  end

  defn grad_window_max_pool(x) do
    grad(x, fn z -> z |> Nx.window_max({1, 1, 2, 2}, strides: [1, 1, 2, 2]) |> Nx.sum() end)
  end

  defn grad_window_avg_pool(x) do
    grad(x, fn z ->
      z
      |> Nx.window_sum({1, 1, 2, 2}, strides: [1, 1, 2, 2])
      |> Nx.divide(4.0)
      |> Nx.sum()
    end)
  end

  # -------------------- Training step --------------------
  # Verbatim copies from Emily.TrainingHelper.

  defn block_forward(params, x, scale) do
    q = Nx.dot(x, params.wq)
    k = Nx.dot(x, params.wk)
    v = Nx.dot(x, params.wv)
    logits = Nx.dot(q, Nx.transpose(k)) * scale
    attn = softmax_last(logits)
    attended = Nx.dot(attn, v) |> Nx.dot(params.wo)
    h = x + attended

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

  # -------------------- Input builders --------------------

  defp det_weights(shape, seed) do
    size = shape |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

    Nx.iota({size}, type: {:f, 32}, backend: Nx.BinaryBackend)
    |> Nx.multiply(0.7)
    |> Nx.add(seed * 7.1)
    |> Nx.sin()
    |> Nx.multiply(0.3)
    |> Nx.reshape(shape)
  end

  defp fixed_inputs(:grad_sum_op), do: [det_weights({3, 4}, 1)]
  defp fixed_inputs(:grad_dot_left), do: [det_weights({3, 4}, 2), det_weights({4, 5}, 3)]
  defp fixed_inputs(:grad_reshape_transpose), do: [det_weights({3, 4}, 4)]
  defp fixed_inputs(:grad_broadcast), do: [det_weights({3}, 5)]

  defp fixed_inputs(:grad_gather) do
    [
      det_weights({4, 5}, 6),
      Nx.tensor([[0, 1], [2, 3], [1, 0]], type: {:s, 32}, backend: Nx.BinaryBackend)
    ]
  end

  defp fixed_inputs(:grad_indexed_add) do
    [
      det_weights({3, 4}, 7),
      Nx.tensor([[0, 1], [2, 3], [1, 0]], type: {:s, 32}, backend: Nx.BinaryBackend),
      Nx.iota({3}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.add(1.0)
    ]
  end

  defp fixed_inputs(:grad_gather_dot_softmax) do
    [
      det_weights({4, 6}, 8),
      Nx.tensor([[0], [2], [1]], backend: Nx.BinaryBackend),
      Nx.iota({6, 5}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(30.0)
    ]
  end

  defp fixed_inputs(:grad_attention) do
    [
      det_weights({3, 4}, 9),
      Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0),
      Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0),
      Nx.iota({4, 4}, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.divide(16.0),
      Nx.tensor(0.5, type: {:f, 32}, backend: Nx.BinaryBackend)
    ]
  end

  defp fixed_inputs(:grad_window_sum), do: [det_weights({2, 3, 4, 4}, 10)]
  defp fixed_inputs(:grad_window_max_pool), do: [det_weights({2, 3, 4, 4}, 11)]
  defp fixed_inputs(:grad_window_avg_pool), do: [det_weights({2, 3, 4, 4}, 12)]

  defp grad_function(:grad_sum_op), do: &grad_sum_op/1
  defp grad_function(:grad_dot_left), do: &grad_dot_left/2
  defp grad_function(:grad_reshape_transpose), do: &grad_reshape_transpose/1
  defp grad_function(:grad_broadcast), do: &grad_broadcast/1
  defp grad_function(:grad_gather), do: &grad_gather/2
  defp grad_function(:grad_indexed_add), do: &grad_indexed_add/3
  defp grad_function(:grad_gather_dot_softmax), do: &grad_gather_dot_softmax/3
  defp grad_function(:grad_attention), do: &grad_attention/5
  defp grad_function(:grad_window_sum), do: &grad_window_sum/1
  defp grad_function(:grad_window_max_pool), do: &grad_window_max_pool/1
  defp grad_function(:grad_window_avg_pool), do: &grad_window_avg_pool/1

  @zoo [
    :grad_sum_op,
    :grad_dot_left,
    :grad_reshape_transpose,
    :grad_broadcast,
    :grad_gather,
    :grad_indexed_add,
    :grad_gather_dot_softmax,
    :grad_attention,
    :grad_window_sum,
    :grad_window_max_pool,
    :grad_window_avg_pool
  ]

  # -------------------- Generator --------------------

  def generate do
    Nx.default_backend(EXLA.Backend)

    goldens = generate_zoo()
    block = generate_block_step()

    # Resolve relative to the script's own location.
    script_dir = Path.dirname(Path.expand(__ENV__.file))
    out = Path.join(script_dir, "../test/support/exla_golden_data.ex") |> Path.expand()
    write_module(goldens, block, out)
  end

  defp generate_zoo do
    for name <- @zoo, into: %{} do
      inputs = fixed_inputs(name)
      fun = grad_function(name)
      result = Nx.Defn.jit_apply(fun, inputs, compiler: EXLA)
      result_bin = Nx.backend_transfer(result, Nx.BinaryBackend)

      {name, %{
        expected: Nx.to_flat_list(result_bin),
        shape: result_bin.shape,
        type: result_bin.type
      }}
    end
  end

  defp generate_block_step do
    embed = 16
    ff = 32
    seq = 8
    lr_val = 0.1
    scale_val = 1.0 / :math.sqrt(embed)

    params = init_block({embed, ff}, 0)
    {x, y} = block_batch({seq, embed})
    lr = Nx.tensor(lr_val, type: {:f, 32}, backend: Nx.BinaryBackend)
    scale = Nx.tensor(scale_val, type: {:f, 32}, backend: Nx.BinaryBackend)

    {new_params, loss} =
      Nx.Defn.jit_apply(
        &block_step_with_loss/5,
        [params, x, y, lr, scale],
        compiler: EXLA
      )

    loss_f = loss |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number()

    param_goldens =
      for key <- [:wq, :wk, :wv, :wo, :w_ff1, :b_ff1, :w_ff2, :b_ff2], into: %{} do
        t = Nx.backend_transfer(new_params[key], Nx.BinaryBackend)
        {key, %{expected: Nx.to_flat_list(t), shape: t.shape, type: t.type}}
      end

    %{loss: loss_f, params: param_goldens}
  end

  defp write_module(goldens, block, path) do
    i = fn list -> inspect(list, limit: :infinity) end

    lines = [
      ~s|defmodule Emily.ExlaGoldenData do|,
      ~s|  @moduledoc \"\"\"|,
      ~s|  EXLA-produced golden gradient values for M13 grad conformance.|,
      ~s||,
      ~s|  Generated by `elixir bench/exla_golden_gen.exs` with EXLA #{Application.spec(:exla, :vsn)}|,
      ~s|  (CPU backend). Regenerate when the grad zoo or its fixed inputs change.|,
      ~s||,
      ~s|  Generated: #{Date.utc_today()}|,
      ~s|  EXLA version: #{Application.spec(:exla, :vsn)}|,
      ~s|  Backend: EXLA (CPU)|,
      ~s|  \"\"\"|,
      ~s||
    ]

    zoo_lines =
      Enum.flat_map(goldens, fn {name, data} ->
        [
          ~s|  def golden(#{inspect(name)}) do|,
          ~s|    %{|,
          ~s|      expected: #{i.(data.expected)},|,
          ~s|      shape: #{inspect(data.shape)},|,
          ~s|      type: #{inspect(data.type)}|,
          ~s|    }|,
          ~s|  end|,
          ~s||
        ]
      end)

    block_param_lines =
      Enum.flat_map(block.params, fn {key, data} ->
        [
          ~s|        #{key}: %{|,
          ~s|          expected: #{i.(data.expected)},|,
          ~s|          shape: #{inspect(data.shape)},|,
          ~s|          type: #{inspect(data.type)}|,
          ~s|        },|
        ]
      end)

    block_lines = [
      ~s|  def block_step_golden do|,
      ~s|    %{|,
      ~s|      loss: #{inspect(block.loss)},|,
      ~s|      params: %{|
    ] ++ block_param_lines ++ [
      ~s|      }|,
      ~s|    }|,
      ~s|  end|,
      ~s|end|
    ]

    content = Enum.join(lines ++ zoo_lines ++ block_lines, "\n") <> "\n"
    File.write!(path, content)
    IO.puts(:stderr, "Wrote #{byte_size(content)} bytes to #{path}")
  end

  defp init_block({embed, ff}, seed) do
    %{
      wq: det_weights({embed, embed}, seed * 31 + 1),
      wk: det_weights({embed, embed}, seed * 31 + 2),
      wv: det_weights({embed, embed}, seed * 31 + 3),
      wo: det_weights({embed, embed}, seed * 31 + 4),
      w_ff1: det_weights({embed, ff}, seed * 31 + 5),
      b_ff1: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: Nx.BinaryBackend), {ff}),
      w_ff2: det_weights({ff, embed}, seed * 31 + 6),
      b_ff2: Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}, backend: Nx.BinaryBackend), {embed})
    }
  end

  defp block_batch({seq, embed}) do
    x = det_weights({seq, embed}, 777)
    y = det_weights({seq, embed}, 888)
    {x, y}
  end
end

GoldenGen.generate()
