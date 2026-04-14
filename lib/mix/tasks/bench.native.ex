defmodule Mix.Tasks.Bench.Native do
  @moduledoc """
  Build and run the standalone C++ microbenchmarks under `bench/native/`.

  These benchmarks link directly against the vendored MLX, so they isolate
  MLX/Metal performance from any BEAM/Nx overhead. They are used to de-risk
  backend features — e.g. M6 (`mlx::core::compile` wrapping) uses
  `compile_microbench.cpp` to check the ≥20% gate before committing to the
  full Emily.Compiler integration.

  ## Usage

      mix bench.native [-- <args passed to the bench binary>]

  Example:

      mix bench.native -- --warmup 20 --iters 500

  The task invokes `make bench-native` with the same MLX/cache env vars
  `elixir_make` uses when building the NIF, so it reuses the existing
  MLX prebuilt without a second fetch.
  """

  use Mix.Task

  @shortdoc "Build and run C++ microbenchmarks under bench/native/"

  @impl Mix.Task
  def run(args) do
    # Ensure MLX prebuilt is fetched. The :emily_mlx compiler is an alias
    # that runs `fetch_mlx/1`; `Mix.Project.compile/1` would run the full
    # NIF build too, which we don't need for a standalone bench. Invoke
    # the alias directly.
    Mix.Task.run("compile.emily_mlx", [])

    # Ask the mix project for its make_env — same map `elixir_make` uses.
    env =
      Mix.Project.config()
      |> Keyword.fetch!(:make_env)
      |> case do
        f when is_function(f, 0) -> f.()
        m when is_map(m) -> m
      end
      |> Enum.to_list()

    make = System.get_env("MAKE") || "make"

    bench_args =
      case args do
        ["--" | rest] -> Enum.join(rest, " ")
        [] -> ""
        rest -> Enum.join(rest, " ")
      end

    cmd_args = ["bench-native"]

    cmd_args =
      if bench_args == "",
        do: cmd_args,
        else: cmd_args ++ ["BENCH_NATIVE_ARGS=" <> bench_args]

    Mix.shell().info("Running: #{make} #{Enum.join(cmd_args, " ")}")

    {_out, status} =
      System.cmd(make, cmd_args, env: env, into: IO.stream(:stdio, :line), stderr_to_stdout: true)

    if status != 0 do
      Mix.raise("bench.native exited with status #{status}")
    end
  end
end
