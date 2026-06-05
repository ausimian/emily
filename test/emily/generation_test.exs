defmodule Emily.GenerationTest do
  @moduledoc """
  CM9 — the model-agnostic decode-loop driver. A tiny hand-rolled
  shape-stable decoder (embedding → dynamic KV write at `offset` → length
  mask → context sum → tied-ish head) exercises the driver end-to-end: the
  native single-NIF forward must produce token ids bit-identical to the
  Evaluator path, and the loop's stop/streaming behaviour must hold.

  No real model / gem_chat dependency — Emily supplies only the mechanism.
  """
  use ExUnit.Case, async: true

  @v 8
  @h 4
  @l 6

  # A minimal shape-stable per-token forward: `fn token, offset, cache,
  # params -> {logits, cache} end`. `offset` is a runtime s32 scalar, so one
  # compiled program serves every position (the KV write is a dynamic
  # put_slice and the window is masked by length, never resized). Weights
  # are passed via `params` (a required arg, not a closure).
  defp tiny_decoder do
    params = %{
      embed: Nx.divide(Nx.iota({@v, @h}, type: :f32, backend: Emily.Backend), 10.0),
      head: Nx.divide(Nx.iota({@h, @v}, type: :f32, backend: Emily.Backend), 7.0)
    }

    cache0 = Nx.broadcast(Nx.tensor(0.0, backend: Emily.Backend), {1, @l, @h})

    forward = fn token, offset, cache, params ->
      x = params.embed |> Nx.take(token) |> Nx.reshape({1, 1, @h})
      cache = Nx.put_slice(cache, [0, offset, 0], x)

      maskf =
        Nx.iota({@l}, type: :s32)
        |> Nx.less_equal(offset)
        |> Nx.as_type(:f32)
        |> Nx.reshape({1, @l, 1})

      ctx = cache |> Nx.multiply(maskf) |> Nx.sum(axes: [1])
      logits = ctx |> Nx.dot([1], params.head, [0]) |> Nx.reshape({@v})
      {logits, cache}
    end

    {forward, cache0, params}
  end

  @native [compiler: Emily.Compiler, native: true, native_fallback: :raise]
  @eval [compiler: Emily.Compiler]

  test "greedy decode under the native compiler matches the evaluator bit-for-bit" do
    {fwd, cache0, params} = tiny_decoder()
    base = [cache: cache0, params: params, first_token: 1, offset: 0, max_new_tokens: 6]

    native = Emily.Generation.stream(fwd, base ++ [defn_options: @native])
    eval = Emily.Generation.stream(fwd, base ++ [defn_options: @eval])

    assert native == eval
    assert length(native) == 6
    assert Enum.all?(native, &(&1 in 0..(@v - 1)))
  end

  test "stops at an eos token (the stop token is included)" do
    {fwd, cache0, params} = tiny_decoder()
    base = [cache: cache0, params: params, first_token: 1, offset: 0, max_new_tokens: 6]

    # Whatever the first generated token is, make it the eos and re-run:
    # the loop must emit exactly that token and stop.
    [first | _] = Emily.Generation.stream(fwd, base)
    assert Emily.Generation.stream(fwd, base ++ [eos: [first]]) == [first]
  end

  test "streams each generated token via :on_token, in order" do
    {fwd, cache0, params} = tiny_decoder()
    parent = self()

    out =
      Emily.Generation.stream(fwd,
        cache: cache0,
        params: params,
        first_token: 1,
        offset: 0,
        max_new_tokens: 4,
        on_token: fn id -> send(parent, {:tok, id}) end
      )

    streamed = for _ <- out, do: receive(do: ({:tok, id} -> id))
    assert streamed == out
  end

  test "max_new_tokens: 0 generates nothing" do
    {fwd, cache0, params} = tiny_decoder()

    assert Emily.Generation.stream(fwd,
             cache: cache0,
             params: params,
             first_token: 1,
             max_new_tokens: 0
           ) == []
  end
end
