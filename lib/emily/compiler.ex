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
  after option validation. There is no external JIT cache beyond the
  closure `Nx.Defn.compile/3` already returns: Bumblebee and
  `Nx.Serving` hold that closure on warmup, so subsequent calls skip
  the walk.

  The compiler does not wrap `mlx::core::compile`. The bench harness
  under `bench/native/` measured the fusion win at <1.2× on
  transformer-shaped workloads — below the threshold that justified
  the integration cost.

  ## Options

    * `:device` — `:gpu` (default) or `:cpu`. Forwarded to
      `Emily.Backend` via the `__to_backend__/1` callback.
    * `:hooks`, `:debug_options`, `:garbage_collect` — passed through
      to `Nx.Defn.Evaluator` unchanged. See its moduledoc.
    * `:max_concurrency` — accepted for `Nx.Serving` compatibility,
      but multi-partition serving is rejected because MLX kernel
      dispatch isn't thread-safe. Pass `1` (the default) to silence.
      For concurrent inference see `Emily.Stream`.

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

  alias Nx.Defn.Evaluator

  @valid_opts [:device, :hooks, :debug_options, :garbage_collect, :max_concurrency]

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    opts = validate_opts!(opts)
    Evaluator.__jit__(key, vars, fun, args_list, opts)
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    opts = validate_opts!(opts)
    Evaluator.__compile__(key, vars, fun, opts)
  end

  @impl true
  def __partitions_options__(opts) do
    opts = validate_opts!(opts)

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
    opts = validate_opts!(opts)
    {Emily.Backend, [device: Keyword.get(opts, :device, :gpu)]}
  end

  defp validate_opts!(opts) do
    case Enum.reject(Keyword.keys(opts), &(&1 in @valid_opts)) do
      [] ->
        opts

      unknown ->
        raise ArgumentError,
              "Emily.Compiler received unknown option(s): #{inspect(unknown)}. " <>
                "Valid options: #{inspect(@valid_opts)}"
    end
  end
end
