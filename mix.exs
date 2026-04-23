defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.2.2"
  @source_url "https://github.com/ausimian/emily"

  # MLX pin. Drives the git tag the `:mlx_src` dep is cloned at (see
  # `deps/0`) and the per-variant cache dir layout. Bump in lockstep with
  # the submodule ref; CI's `release-nif.yml` rebuilds the NIF against
  # whatever this resolves to.
  @mlx_version "0.31.2"

  require Logger

  def project do
    [
      app: @app,
      version: @version,
      description: "Elixir bindings and Nx backend for Apple MLX",
      source_url: @source_url,
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),
      compilers: [:emily_mlx, :elixir_make] ++ Mix.compilers(),
      make_env: &make_env/0,
      make_args: ["-j#{System.schedulers_online()}"],
      test_coverage: test_coverage(),
      dialyzer: dialyzer(),
      docs: docs(),
      package: package()
    ]
  end

  def cli do
    [preferred_envs: [docs: :docs, "hex.publish": :docs, precommit: :test]]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {Emily.Application, []}
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  # Emily.Native is pure NIF stubs — :erlang.load_nif/2 patches the bytecode
  # at load time, so the stub bodies never run and cover reports 0% on them.
  # Excluding the module drops that artefact and lets the remaining Elixir
  # coverage number mean something.
  defp test_coverage, do: [ignore_modules: [Emily.Native]]

  defp deps do
    [
      {:elixir_make, "~> 0.9"},
      {:fine, "~> 0.1"},
      {:nx, "~> 0.10"},
      # Bumblebee + Axon are declared `optional: true` because the
      # only Emily module that touches either — `Emily.Bumblebee.FastKernels`
      # — is wrapped in a `Code.ensure_loaded?/1` gate and elides when
      # they are absent. Consumers who want the shim pull both in
      # themselves; everyone else gets a clean build with no
      # Bumblebee/Axon/Tokenizers in their deps tree.
      #
      # Crucially `optional: true` without an `only:` env filter is
      # what makes the gate actually work. The optional relationship
      # must be visible to Mix in the consumer's build env so
      # Axon/Bumblebee get compiled *before* Emily — otherwise
      # `Code.ensure_loaded?(Bumblebee.Layers)` at Emily's compile
      # time returns false and the shim elides even when the consumer
      # has both deps declared.
      {:bumblebee, "~> 0.6", optional: true},
      {:tokenizers, "~> 0.5", optional: true},
      {:axon, "~> 0.7", optional: true},
      # `scidata` loads MNIST / CIFAR / etc. for the `:training_full`
      # opt-in convergence canary (M9). Kept test-only — Emily itself
      # doesn't depend on dataset loading.
      {:scidata, "~> 0.1", only: :test},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev], runtime: false},
      {:ex_doc, "~> 0.34", only: :docs, runtime: false},
      {:publisho, "~> 1.0", only: :dev, runtime: false},
      # MLX source tree for in-repo/CI source builds of libmlx + the
      # Metal shader library. Cloned by `mix deps.get` and consumed by
      # `scripts/build-mlx.sh` via the `compile.emily_mlx` alias below.
      # Hex consumers never see this — they receive a precompiled NIF,
      # so MLX source isn't needed at their build time.
      {:mlx_src,
       git: "https://github.com/ml-explore/mlx.git",
       tag: "v#{@mlx_version}",
       app: false,
       compile: false,
       only: [:dev, :test]}
    ]
  end

  defp dialyzer do
    [
      plt_add_apps: [:mix, :ex_unit],
      plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
      plt_core_path: "priv/plts/core.plt",
      flags: [:error_handling, :unknown, :unmatched_returns, :extra_return],
      ignore_warnings: ".dialyzer_ignore.exs"
    ]
  end

  defp aliases do
    [
      precommit: [
        "compile --warnings-as-errors",
        "deps.unlock --unused",
        "format",
        "credo --strict",
        "test"
      ],
      "compile.emily_mlx": &build_mlx/1
    ]
  end

  defp docs do
    [
      main: "readme",
      source_url_pattern: "#{@source_url}/blob/#{@version}/%{path}#L%{line}",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "notebooks/distilbert_qa.livemd",
        "notebooks/qwen3_quantized.livemd",
        "notebooks/mnist_training.livemd",
        "notebooks/whisper_transcription.livemd",
        "notebooks/fast_kernels.livemd"
      ],
      groups_for_extras: [
        README: ~r{README.md},
        Notebooks: ~r{^notebooks/}
      ],
      groups_for_modules: [
        Core: [Emily, Emily.Backend, Emily.Compiler],
        Concurrency: [Emily.Stream],
        Quantization: [
          Emily.Quantization,
          Emily.Quantization.Layers,
          Emily.QuantizedWeight
        ],
        Training: [Emily.MixedPrecision, Emily.MixedPrecision.LossScaler],
        Performance: [Emily.Fast, Emily.Bumblebee.FastKernels],
        Observability: [Emily.Telemetry]
      ]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib c_src Makefile mix.exs config README.md CHANGELOG.md LICENSE)
    ]
  end

  # ---------- MLX source build (in-repo / CI) ----------

  defp make_env do
    dir = mlx_install_dir()

    %{
      "MLX_DIR" => dir,
      "MLX_INCLUDE_DIR" => Path.join(dir, "include"),
      "MLX_LIB_DIR" => Path.join(dir, "lib"),
      "FINE_INCLUDE_DIR" => Fine.include_dir(),
      "EMILY_CACHE_DIR" => cache_dir(),
      "EMILY_VERSION" => @version
    }
  end

  defp cache_dir do
    case System.get_env("EMILY_CACHE") do
      nil -> :filename.basedir(:user_cache, ~c"emily") |> to_string()
      dir -> Path.expand(dir)
    end
  end

  defp mlx_variant do
    case Application.get_env(:emily, :variant, :aot) do
      :aot -> "aot"
      :jit -> "jit"
      other -> Mix.raise("Invalid :emily variant #{inspect(other)}. Expected :aot or :jit.")
    end
  end

  defp mlx_install_dir,
    do: Path.join(cache_dir(), "mlx-#{@mlx_version}-#{mlx_variant()}")

  defp arch_tag do
    case {:os.type(), :erlang.system_info(:system_architecture) |> to_string()} do
      {{:unix, :darwin}, "aarch64" <> _} ->
        "arm64"

      {{:unix, :darwin}, "x86_64" <> _} ->
        Mix.raise("""
        x86_64 macOS is not supported for MLX #{@mlx_version}.
        Apple Silicon is required.
        """)

      {os, arch} ->
        Mix.raise("""
        Emily's MLX build is macOS-only; cannot build on
        #{inspect(os)} / #{arch}.
        """)
    end
  end

  defp build_mlx(args) do
    _ = arch_tag()
    dir = mlx_install_dir()

    if "--force" in args do
      File.rm_rf!(dir)
    end

    if File.dir?(dir) do
      {:ok, []}
    else
      build_mlx_from_source!(dir)
      {:ok, []}
    end
  end

  defp build_mlx_from_source!(install_dir) do
    mlx_src = Path.expand("deps/mlx_src", File.cwd!())

    unless File.dir?(mlx_src) do
      Mix.raise("""
      MLX source not found at #{mlx_src}.
      Run `mix deps.get` to clone the `:mlx_src` git dep.
      """)
    end

    script = Path.expand("scripts/build-mlx.sh", File.cwd!())
    jit_flag = if mlx_variant() == "jit", do: "1", else: "0"

    File.mkdir_p!(cache_dir())

    Mix.shell().info("Building MLX #{@mlx_version} (#{mlx_variant()}) from source")

    port_opts = [
      :binary,
      :exit_status,
      :stderr_to_stdout,
      {:args, [mlx_src, @mlx_version, jit_flag, install_dir]}
    ]

    port = Port.open({:spawn_executable, String.to_charlist(script)}, port_opts)

    case stream_port(port) do
      0 ->
        :ok

      code ->
        File.rm_rf(install_dir)
        Mix.raise("MLX source build failed (exit #{code})")
    end
  end

  defp stream_port(port) do
    receive do
      {^port, {:data, bin}} ->
        IO.write(bin)
        stream_port(port)

      {^port, {:exit_status, code}} ->
        code
    end
  end
end
