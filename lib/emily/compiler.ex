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
    :native
  ]

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    opts = take_known_opts(opts)

    if Keyword.get(opts, :native, false) do
      compile_native(vars, fun).(args_list)
    else
      Evaluator.__jit__(key, vars, fun, args_list, opts)
    end
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    opts = take_known_opts(opts)

    if Keyword.get(opts, :native, false) do
      compile_native(vars, fun)
    else
      Evaluator.__compile__(key, vars, fun, opts)
    end
  end

  # The single-NIF compiled path (CM1+): trace the function into an
  # Nx.Defn.Expr, lower it to a flat IR once, compile it into a `Program`
  # resource (captured in this closure), and replay the whole graph in
  # one NIF call per invocation. Op coverage is still partial — an
  # unsupported op raises in `Emily.IR.lower/1` (no silent fallback).
  defp compile_native(vars, fun) do
    expr = fun.(vars)

    {template, leaves_rev} =
      Composite.traverse(expr, [], fn leaf, acc -> {Nx.to_template(leaf), [leaf | acc]} end)

    program = leaves_rev |> Enum.reverse() |> IR.lower() |> Program.compile()

    fn [params] ->
      worker = Emily.MlxStream.default_worker()
      # Params arrive as zero-arity realizer funs in slot order (see
      # Nx.Defn.Evaluator's `:parameter` handling); realize each to a
      # native ref. `{:input, i}` in the IR indexes this list.
      input_refs = Enum.map(params, fn p -> p.() |> Nx.to_tensor() |> native_ref() end)
      out_refs = Program.eval(worker, program, input_refs)
      [reassemble(template, out_refs)]
    end
  end

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
