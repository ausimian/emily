defmodule Mix.Tasks.Clang.Tidy do
  @moduledoc """
  Run clang-tidy over the C++ NIF sources in `c_src/`.

  Unlike `make cppcheck`, clang-tidy compiles each translation unit, so it
  needs the MLX / Fine / ERTS headers and the exact build flags. This task
  supplies the same env `elixir_make` uses when building the NIF — reusing
  the already-built (cached) MLX rather than fetching a second copy — and
  then invokes the `clang-tidy` Makefile target, which analyses the NIF
  sources with the build's `$(CXXFLAGS)`.

  ## Usage

      mix clang.tidy

  Requires clang-tidy on `PATH` (`brew install llvm`); point at a specific
  binary with `CLANG_TIDY=/path/to/clang-tidy`. The enabled checks and the
  header filter live in the repo-root `.clang-tidy`.
  """

  use Mix.Task

  @shortdoc "Run clang-tidy over the C++ NIF sources"

  @impl Mix.Task
  def run(_args) do
    # Ensure MLX is available (its headers are what clang-tidy parses). The
    # :emily_mlx compiler alias reuses an existing install and only builds
    # from source on a cold cache — same as `mix bench.native`.
    Mix.Task.run("compile.emily_mlx", [])

    # Ask the mix project for its make_env — the same map `elixir_make`
    # passes to `make` — and add ERTS_INCLUDE_DIR, which elixir_make sets
    # itself at build time (so make_env/0 omits it) but a standalone `make`
    # invocation does not get.
    env =
      Mix.Project.config()
      |> Keyword.fetch!(:make_env)
      |> case do
        f when is_function(f, 0) -> f.()
        m when is_map(m) -> m
      end
      |> Map.put("ERTS_INCLUDE_DIR", erts_include_dir())
      |> Enum.to_list()

    make = System.get_env("MAKE") || "make"

    Mix.shell().info("Running: #{make} clang-tidy")

    {_out, status} =
      System.cmd(make, ["clang-tidy"],
        env: env,
        into: IO.stream(:stdio, :line),
        stderr_to_stdout: true
      )

    if status != 0 do
      Mix.raise("clang.tidy reported findings (exit #{status})")
    end
  end

  # Mirror how elixir_make derives ERTS_INCLUDE_DIR: the Erlang headers
  # (erl_nif.h, reached through fine.hpp) live under the OTP install. Honour
  # an explicit override if one is already exported.
  defp erts_include_dir do
    System.get_env("ERTS_INCLUDE_DIR") ||
      Path.join([
        to_string(:code.root_dir()),
        "erts-#{:erlang.system_info(:version)}",
        "include"
      ])
  end
end
