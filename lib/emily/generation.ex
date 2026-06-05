defmodule Emily.Generation do
  @moduledoc """
  A minimal, model-agnostic **decode-loop driver** for autoregressive
  generation on Emily's native compiler.

  Emily is an Nx backend, not a model library — so this module supplies
  only the *mechanism*. It JIT-compiles a caller-supplied **shape-stable**
  per-token forward with the single-NIF native compiler (see
  `Emily.Compiler`) and drives the token loop from Elixir: offset
  bookkeeping, KV-cache threading, the stop conditions, next-token
  selection, and streaming. The caller owns the *model*: it provides the
  forward and the (pre-filled) cache.

  This is the "loop in Elixir" half of the generation story — it preserves
  per-token streaming and host-side control. (`Bumblebee.Text.generation`
  compiles its own `defn while` loop instead; that path is handled by the
  native `while` opcode, not this driver.)

  ## The forward contract

  The forward is an arity-4 function `fn token, offset, cache, params ->
  {logits, cache} end`, traceable by `Nx.Defn` and **shape-stable**: its
  argument and result shapes must not depend on the runtime value of
  `offset`, so a single compiled program serves every position.
  Concretely:

    * `token`  — an `s32` `{1}` tensor (the id to decode at this step),
    * `offset` — an `s32` scalar tensor (the absolute position; thread it
      as a runtime input, e.g. a dynamic `Nx.put_slice` into a fixed-size
      KV buffer plus a length mask, rather than a growing slice),
    * `cache`  — an `Nx.Container` of fixed-shape KV buffers,
    * `params` — an `Nx.Container` of the model weights,

  returning `{logits, cache}` where `logits` is the last position's logit
  vector and `cache` has the same structure/shapes as the input.

  The driver does **not** bound `offset` against the cache window — sizing
  `offset + max_new_tokens` to fit the fixed KV buffer is the caller's
  responsibility (overflowing it silently corrupts the cache via the
  out-of-bounds `put_slice`, it does not raise).

  `params` is a *required argument* rather than a closure on purpose: Nx
  rejects mixing closed-over `Emily.Backend` tensors with the traced
  `Nx.Defn.Expr`, and passing them as an argument also hands their refs to
  the compiled program zero-copy (captured once).

  ## Example

      # `forward`, `cache0`, and `params` come from your model.
      tokens =
        Emily.Generation.stream(forward,
          cache: cache0,
          params: params,
          first_token: bos_id,
          offset: prompt_len,
          max_new_tokens: 64,
          eos: [eos_id],
          on_token: fn id -> send(self(), {:token, id}) end
        )

  Returns the list of generated token ids (including the stop token, if
  one is hit). `:select` defaults to greedy `argmax`; pass your own
  `fn logits -> token_tensor end` for sampling.
  """

  @default_defn_options [compiler: Emily.Compiler, native: true]

  @doc """
  Greedy next-token selector: `argmax` over the vocabulary (last) axis.
  """
  @spec greedy(Nx.Tensor.t()) :: Nx.Tensor.t()
  def greedy(logits), do: Nx.argmax(logits, axis: -1)

  @doc """
  Drive an autoregressive decode loop over a shape-stable `forward`.

  ## Options

    * `:cache` (required) — the initial (pre-filled) KV-cache container.
    * `:params` (required) — the model weights container, passed to the
      forward each step.
    * `:first_token` (required) — the first token id to decode.
    * `:max_new_tokens` (required) — the maximum number of tokens to emit.
    * `:offset` — the starting absolute position (default `0`).
    * `:eos` — a stop token id or list of ids (default `[]`).
    * `:select` — `fn logits -> token_tensor end` (default `greedy/1`).
    * `:on_token` — `fn token_id -> any end`, called with each generated
      id as it is produced (default no-op).
    * `:defn_options` — options for `Nx.Defn.compile/3` (default
      `#{inspect(@default_defn_options)}`). Override to disable native
      compilation or pick a different compiler.

  Returns the list of generated token ids.
  """
  @spec stream(
          (Nx.Tensor.t(), Nx.Tensor.t(), Nx.Container.t(), Nx.Container.t() ->
             {Nx.Tensor.t(), Nx.Container.t()}),
          keyword()
        ) :: [integer()]
  def stream(forward, opts) when is_function(forward, 4) do
    cache = Keyword.fetch!(opts, :cache)
    params = Keyword.fetch!(opts, :params)
    first = Keyword.fetch!(opts, :first_token)
    max_new = Keyword.fetch!(opts, :max_new_tokens)
    offset = Keyword.get(opts, :offset, 0)
    eos = opts |> Keyword.get(:eos, []) |> List.wrap() |> MapSet.new()
    select = Keyword.get(opts, :select, &greedy/1)
    on_token = Keyword.get(opts, :on_token, fn _ -> :ok end)
    defn_options = Keyword.get(opts, :defn_options, @default_defn_options)

    # Compile the forward once: `offset` and the token id are runtime
    # inputs, so the program is reused across every step (and every request
    # with the same cache/param shapes). The concrete `cache`/`params`
    # supply the container templates.
    step =
      Nx.Defn.compile(
        forward,
        [Nx.template({1}, :s32), Nx.template({}, :s32), cache, params],
        defn_options
      )

    # The loop-invariant context (compiled step, params, stop set, selector,
    # streaming callback) travels as one map so only the changing state —
    # current token, cache, offset, budget, accumulator — is threaded.
    ctx = %{step: step, params: params, eos: eos, select: select, on_token: on_token}
    loop(ctx, first, cache, offset, max_new, [])
  end

  defp loop(_ctx, _cur, _cache, _offset, budget, acc) when budget <= 0,
    do: Enum.reverse(acc)

  defp loop(ctx, cur, cache, offset, budget, acc) do
    token = Nx.tensor([cur], type: :s32, backend: Emily.Backend)
    offset_t = Nx.tensor(offset, type: :s32, backend: Emily.Backend)

    {logits, cache} = ctx.step.(token, offset_t, cache, ctx.params)
    next = ctx.select.(logits) |> Nx.to_number()
    ctx.on_token.(next)
    acc = [next | acc]

    if MapSet.member?(ctx.eos, next) do
      Enum.reverse(acc)
    else
      loop(ctx, next, cache, offset + 1, budget - 1, acc)
    end
  end
end
