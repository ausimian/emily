defmodule Emily.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` implementation that runs `defn` computations on
  `Emily.Backend`.

  The compiler walks `Nx.Defn.Expr` in Elixir and dispatches each node
  through the active backend — exactly what `Nx.Defn.Evaluator` already
  does — with two adjustments specific to Emily:

    * `__to_backend__/1` returns `{Emily.Backend, [device: …]}` so
      `Nx.Defn.to_backend/1` (and the callers that consult it, including
      `Nx.Serving`) allocate inputs and outputs on Emily rather than the
      process-default backend.
    * `__partitions_options__/1` always returns a single partition.
      MLX's Metal runtime was historically unsafe for concurrent kernel
      dispatch from multiple OS threads. `:max_concurrency` is accepted
      for API compatibility with `Nx.Serving` but capped at 1. For
      concurrent inference on a shared model use `Emily.Stream`.

  ## Public API

  Users do not call this module directly. Install it as the default
  compiler and `Nx.Serving` / Bumblebee picks it up:

      Nx.Defn.global_default_options(compiler: Emily.Compiler)

  Or attach it per-call:

      Nx.Defn.jit(&my_fn/1, compiler: Emily.Compiler).(input)

  The four callbacks on `Nx.Defn.Compiler` (`__jit__/5`,
  `__compile__/4`, `__partitions_options__/1`, `__to_backend__/1`)
  are invoked by `Nx.Defn` on your behalf.

  ## Design notes

  `__jit__/5` and `__compile__/4` delegate to `Nx.Defn.Evaluator`
  after filtering the option list down to the keys this module
  consumes. There is no external JIT cache beyond the
  closure `Nx.Defn.compile/3` already returns: Bumblebee and
  `Nx.Serving` hold that closure on warmup, so subsequent calls skip
  the walk.

  The compiler does not wrap `mlx::core::compile` by default. The
  single-NIF replay is the load-bearing win (it collapses the per-op
  BEAM↔worker round-trips); `mx::compile` is exposed as an *opt-in*
  compiled eval mode on the program resource, which fuses the
  elementwise runs the replay leaves separate. On a decode-shaped
  transformer block `bench/program_compile.exs` measures ~1.6× over the
  sync replay (kernel-launch + intermediate-memory overhead dominates at
  small sequence lengths, and fusion removes it), at the cost of
  last-few-ULP f32 reassociation and a shape-stability requirement —
  hence opt-in, not the default for the general compiler.

  ## Options

    * `:device` — `:gpu` (default) or `:cpu`. Forwarded to
      `Emily.Backend` via the `__to_backend__/1` callback.
    * `:hooks`, `:debug_options`, `:garbage_collect` — passed through
      to `Nx.Defn.Evaluator` unchanged. See its moduledoc.
    * `:max_concurrency` — accepted for `Nx.Serving` compatibility,
      but multi-partition serving is rejected because MLX kernel
      dispatch isn't thread-safe. Pass `1` (the default) to silence.
      For concurrent inference see `Emily.Stream`.
    * `:batch_keys`, `:cache` — accepted and ignored. `Nx.Serving`
      propagates `:batch_keys` to the compiler via `defn_options` for
      arity-1 serving builders (e.g. `Bumblebee.Audio.speech_to_text_whisper/5`),
      and Bumblebee passes `:cache` through for its own per-scope
      cache suffixing. Neither is used by the Evaluator walk, but
      rejecting them would break those servings.
    * `:native` — `true` (the default) compiles the traced
      `Nx.Defn.Expr` to a flat IR and replays the whole graph in a
      single NIF call per invocation; `false` runs the op-by-op
      Evaluator walk instead. The default is read from
      `config :emily, :native` (itself defaulting to `true`), so
      `config :emily, native: false` opts every defn out of the native
      lane application-wide — e.g. on a memory-constrained host where
      the one-shot compile peak is too large. The per-call option wins
      over the app env, so an explicit `native: false` overrides a
      global `config :emily, native: true` and vice versa. A non-boolean
      raises `ArgumentError`.
    * `:native_fallback` — `:eval` (default) or `:raise`. Controls what
      happens when `native: true` but the expression contains an op or
      construct the IR can't lower yet. `:eval` routes the *whole* defn
      through `Nx.Defn.Evaluator` (each op then dispatches through
      `Emily.Backend`, with its own per-op `via_binary` fallback) and
      fires a one-shot `[:emily, :compiler, :fallback]` event, so
      installing `compiler: Emily.Compiler, native: true` globally is
      safe on any model. `:raise` re-raises the lowering error instead —
      use it in CI to prove a model lowers fully native. The per-call
      option wins over `config :emily, :native_fallback, :eval | :raise`.
    * `:fuse` — `true` evals the compiled program in the
      `mx::compile`'d mode instead of the plain replay. For a while-free
      forward this fuses the elementwise runs the replay leaves separate
      (the CM6 win); for a `Bumblebee.Text.generation` `defn while` it keeps
      the decode loop host-controlled but fuses each loop **body** under
      `mx::compile`, replaying the cached fused callable every token. Defaults
      to `false`; a non-boolean raises `ArgumentError`. Opt-in because the
      fusion reassociates f32 to within a few ULP — logits are not
      bit-identical to the evaluator. Greedy argmax is robust to that drift
      (greedy token ids matched the evaluator in our tests), but the match is
      empirical, not guaranteed: any discrete decision the drift can tip —
      argmax on a near-tie, or a `while` trip count whose condition reads a
      reassociated reduction — diverges once it flips. **Sampling strategies
      diverge from the evaluator under fusion** even with a fixed seed. Only
      the native path consults it, so it is ignored unless `native: true`.

  Any other option is silently dropped. This matches how
  `Nx.Defn.Evaluator` and EXLA handle their own option lists, and is
  the contract higher-level libraries rely on when they forward
  caller-supplied options to the JIT compiler — e.g. `Axon.build/2`,
  whose docs state that "all other options are forwarded to the
  underlying JIT compiler".

  ## Examples

  Process-global installation (typical for `Nx.Serving` / Bumblebee):

      Nx.global_default_backend(Emily.Backend)
      Nx.Defn.global_default_options(compiler: Emily.Compiler)

  Per-call:

      add_one = Nx.Defn.jit(fn x -> Nx.add(x, 1) end, compiler: Emily.Compiler)
      add_one.(Nx.tensor([1.0, 2.0]))
      # => #Nx.Tensor<f32[2] [2.0, 3.0]> on Emily.Backend

  """

  @behaviour Nx.Defn.Compiler

  alias Emily.Backend, as: B
  alias Emily.{IR, Program}
  alias Nx.Defn.{Composite, Evaluator}
  alias Nx.Tensor, as: T

  @valid_opts [
    :device,
    :hooks,
    :debug_options,
    :garbage_collect,
    :max_concurrency,
    :batch_keys,
    :cache,
    :native,
    :native_fallback,
    :fuse
  ]

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    opts = take_known_opts(opts)

    if native?(opts) do
      case build_native(key, vars, fun, opts) do
        {:ok, run} -> run.(args_list)
        :fallback -> Evaluator.__jit__(key, vars, fun, args_list, drop_native_opts(opts))
      end
    else
      Evaluator.__jit__(key, vars, fun, args_list, opts)
    end
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    opts = take_known_opts(opts)

    if native?(opts) do
      case build_native(key, vars, fun, opts) do
        {:ok, run} -> run
        :fallback -> Evaluator.__compile__(key, vars, fun, drop_native_opts(opts))
      end
    else
      Evaluator.__compile__(key, vars, fun, opts)
    end
  end

  # Per-call `:native` opt wins over `config :emily, :native`, defaulting
  # to `true` (the native single-NIF lane). `Keyword.fetch/2` (not
  # `Keyword.get/3`) so an explicit per-call `native: false` overrides a
  # global default, rather than being treated as "unset" and falling
  # through to the app env.
  defp native?(opts) do
    enabled =
      case Keyword.fetch(opts, :native) do
        {:ok, native} -> native
        :error -> Application.get_env(:emily, :native, true)
      end

    case enabled do
      bool when is_boolean(bool) ->
        bool

      other ->
        raise ArgumentError,
              "invalid :native #{inspect(other)}; expected true | false"
    end
  end

  # Build the single-NIF native closure for `fun`, or signal `:fallback`.
  #
  # The Expr trace (`fun.(vars)`) runs *outside* the rescue so a genuine
  # caller error surfaces unchanged; only the lowering + program build is
  # guarded. `Emily.IR.lower/1` raises `ArgumentError` on an op or
  # construct it can't lower yet. Unless `:native_fallback` is `:raise`,
  # we emit a one-shot `[:emily, :compiler, :fallback]` event and return
  # `:fallback`, so the caller routes the whole defn through
  # `Nx.Defn.Evaluator` (which dispatches each op through `Emily.Backend`,
  # with its own per-op `via_binary` fallback). This keeps a global
  # `native: true` install safe on any model.
  @spec build_native(term(), [Nx.Tensor.t()], fun(), keyword()) ::
          {:ok, ([term()] -> [Nx.Tensor.t()])} | :fallback
  defp build_native(key, vars, fun, opts) do
    # Resolve (and validate) the modes up front so a misconfigured
    # `:native_fallback` or `:fuse` raises on every call —
    # including the happy path, and the lowering-failure path — rather than
    # lying dormant until the first lowering failure.
    mode = native_fallback_mode(opts)
    eval_mode = native_eval_mode(opts)

    # The Expr trace runs outside `lower/3`'s guard so a genuine caller
    # error surfaces unchanged.
    expr = fun.(vars)

    {template, leaves_rev} =
      Composite.traverse(expr, [], fn leaf, acc -> {Nx.to_template(leaf), [leaf | acc]} end)

    # The flattened parameter leaves are the true input count and slot order
    # (the closure realises them in this order; `{:input, i}` indexes it).
    # `IR.lower` only counts the parameters it *references*, which undercounts
    # when an input is unused (e.g. the `seed` in greedy generation) — pin
    # `n_inputs` to the real arity so the eval-time input count matches.
    n_inputs = length(Composite.flatten_list(vars))

    case lower(Enum.reverse(leaves_rev), mode, key) do
      {:ok, ir} -> {:ok, replay_closure(template, %{ir | n_inputs: n_inputs}, eval_mode)}
      :fallback -> :fallback
    end
  end

  # Resolve the program eval mode from `:fuse`, validating up front
  # (like `native_fallback_mode/1`) so a non-boolean raises on every native
  # call rather than being silently treated as truthy. `true` -> `:compiled`
  # (the `mx::compile` fusion — and, for a `defn while`, fusing each loop
  # *body* under a host-controlled decode loop; see `Emily.Program.eval`),
  # `false` -> `:sync` (the plain, bit-identical replay). Only the native path
  # calls this, so the option is ignored unless `native: true`.
  defp native_eval_mode(opts) do
    case Keyword.get(opts, :fuse, false) do
      true ->
        :compiled

      false ->
        :sync

      other ->
        raise ArgumentError,
              "invalid :fuse #{inspect(other)}; expected true | false"
    end
  end

  # Lower the output leaves to a flat IR. `Emily.IR.lower/1` is the *only*
  # guarded step: it raises `ArgumentError` on an op or construct it can't
  # lower yet, which we turn into a graceful `:fallback` (or re-raise in
  # `:raise` mode). `Program.compile/1` is deliberately kept outside the
  # rescue (in `replay_closure/3`) — it raises only on malformed IR, i.e. a
  # compiler bug, which must surface loudly rather than be masked as an
  # "unsupported op" fallback.
  defp lower(leaves, mode, key) do
    {:ok, IR.lower(leaves)}
  rescue
    e in ArgumentError ->
      case mode do
        :raise ->
          reraise(e, __STACKTRACE__)

        :eval ->
          Emily.Telemetry.compiler_fallback(key, e)
          :fallback
      end
  end

  # The single-NIF compiled path (CM1+): compile the lowered IR into a
  # `Program` resource (captured in this closure) and replay the whole
  # graph in one NIF call per invocation.
  defp replay_closure(template, ir, eval_mode) do
    program = Program.compile(ir)

    fn [params] ->
      worker = Emily.MlxStream.default_worker()
      # Params arrive as zero-arity realizer funs in slot order (see
      # Nx.Defn.Evaluator's `:parameter` handling); realize each to a
      # native ref. `{:input, i}` in the IR indexes this list.
      input_refs = Enum.map(params, fn p -> p.() |> Nx.to_tensor() |> native_ref() end)
      out_refs = Program.eval(worker, program, input_refs, mode: eval_mode)
      [reassemble(template, out_refs)]
    end
  end

  # Per-call `:native_fallback` opt wins over `config :emily,
  # :native_fallback`, defaulting to `:eval`. `Keyword.fetch/2` (not `||`)
  # so an explicit `native_fallback: false` is rejected, not silently
  # treated as "unset".
  defp native_fallback_mode(opts) do
    mode =
      case Keyword.fetch(opts, :native_fallback) do
        {:ok, m} -> m
        :error -> Application.get_env(:emily, :native_fallback, :eval)
      end

    case mode do
      m when m in [:eval, :raise] ->
        m

      other ->
        raise ArgumentError,
              "invalid :native_fallback #{inspect(other)}; expected :eval | :raise"
    end
  end

  # Strip the Emily-only native knobs before delegating to the Evaluator
  # — it ignores keys it doesn't consume, but handing it `native: true`
  # when we've decided *not* to compile natively would be misleading.
  defp drop_native_opts(opts),
    do: Keyword.drop(opts, [:native, :native_fallback, :fuse])

  defp native_ref(%T{data: %B{ref: r}}), do: r
  defp native_ref(%T{} = t), do: Nx.backend_transfer(t, B).data.ref

  defp reassemble(template, out_refs) do
    {result, []} =
      Composite.traverse(template, out_refs, fn leaf, [ref | rest] ->
        {%{leaf | data: %B{ref: ref}}, rest}
      end)

    result
  end

  @impl true
  def __partitions_options__(opts) do
    opts = take_known_opts(opts)

    case Keyword.get(opts, :max_concurrency, 1) do
      n when n in [nil, 1] ->
        [opts]

      n when is_integer(n) and n > 1 ->
        raise ArgumentError,
              "Emily.Compiler does not support :max_concurrency > 1 directly. " <>
                "Use Emily.Stream.with_stream/2 for per-process streams (one " <>
                "shared model, concurrent Metal command queues), or start " <>
                "multiple Nx.Serving instances behind a pool. " <>
                "See Emily.Stream moduledoc for details. Got: #{n}"
    end
  end

  @impl true
  def __to_backend__(opts) do
    opts = take_known_opts(opts)
    {Emily.Backend, [device: Keyword.get(opts, :device, :gpu)]}
  end

  defp take_known_opts(opts), do: Keyword.take(opts, @valid_opts)
end
