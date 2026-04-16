defmodule Emily.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` implementation that runs `defn` computations on
  `Emily.Backend`.

  The compiler walks `Nx.Defn.Expr` in Elixir and dispatches each node
  through the active backend — exactly what `Nx.Defn.Evaluator` already
  does — with two adjustments specific to Emily:

    * `c:__to_backend__/1` returns `{Emily.Backend, [device: …]}` so
      `Nx.Defn.to_backend/1` (and the callers that consult it, including
      `Nx.Serving`) allocate inputs and outputs on Emily rather than the
      process-default backend.
    * `c:__partitions_options__/1` always returns a single partition.
      MLX's Metal runtime is not safe for concurrent kernel dispatch
      from multiple OS threads (see `test/soak/backend_concurrency_test.exs`
      for the SIGSEGV story); a multi-partition serving would race the
      driver. `:max_concurrency` is accepted for API compatibility with
      `Nx.Serving` but capped at 1.

  ## Why this is so thin

  M5 deliberately avoids two pieces of complexity:

    1. **No external cache.** `__compile__/4` walks the expression once
       and returns a closure that captures the walked plan; the closure
       *is* the cache. Callers that want reuse across invocations use
       `Nx.Defn.compile/3` and hold the returned function (Bumblebee /
       `Nx.Serving` already do this on warmup). The PLAN.md note about
       caching the walk in ETS was rejected once we accounted for the
       per-call ETS deep-copy cost on a Qwen3-sized expression tree.

    2. **No `mlx::core::compile` wrapping.** Lazy evaluation at the
       Backend layer is the shipping story. Wrapping
       `mlx::core::compile` was scoped as M6 and then dropped after a
       pure-C++ microbenchmark showed the fusion win on transformer
       workloads is below the PLAN's 1.20× gate (GPU ~1.04–1.07×; CPU
       regresses). The bench harness remains in `bench/native/` for
       re-measurement against future MLX releases; see
       `bench/compile_microbench.md` for the numbers and reasoning.

  Concretely, `__jit__/5` and `__compile__/4` delegate to
  `Nx.Defn.Evaluator` after option validation. The Evaluator dispatches
  every op via `Nx.Shared.list_impl!/1`, which finds `Emily.Backend`
  whenever the operands carry it — and `c:__to_backend__/1` ensures the
  operands do.

  ## Options

    * `:device` — `:gpu` (default) or `:cpu`. Forwarded to `Emily.Backend`
      via the `c:__to_backend__/1` callback.
    * `:hooks`, `:debug_options`, `:garbage_collect` — passed through to
      `Nx.Defn.Evaluator` unchanged. See its moduledoc.
    * `:max_concurrency` — accepted for `Nx.Serving` compatibility, but
      multi-partition serving is rejected because MLX kernel dispatch
      isn't thread-safe. Pass `1` (the default) to silence. For
      concurrent inference see `Emily.Stream`.
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
