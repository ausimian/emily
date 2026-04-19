defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.1.2"
  @source_url "https://github.com/ausimian/emily"

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
      {:publisho, "~> 1.0", only: :dev, runtime: false}
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
      files:
        ~w(lib c_src patches vendor/mlx/mlx vendor/mlx/cmake vendor/mlx/CMakeLists.txt vendor/mlx/mlx.pc.in vendor/mlx/LICENSE vendor/mlx/ACKNOWLEDGMENTS.md Makefile mix.exs README.md CHANGELOG.md LICENSE)
    ]
  end

  # ---------- MLX from source ----------

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

  @mlx_source_dir Path.expand("vendor/mlx", __DIR__)

  defp mlx_cache_key do
    # In a git checkout the submodule commit is the best cache key.
    # In a Hex install (no .git), hash the top-level CMakeLists.txt.
    case System.cmd("git", ["-C", @mlx_source_dir, "rev-parse", "--short", "HEAD"],
           stderr_to_stdout: true
         ) do
      {hash, 0} -> String.trim(hash)
      _ -> cmake_hash()
    end
  end

  defp cmake_hash do
    @mlx_source_dir
    |> Path.join("CMakeLists.txt")
    |> File.read!()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower)
    |> binary_part(0, 12)
  end

  defp mlx_install_dir do
    Path.join(cache_dir(), "mlx-#{mlx_cache_key()}#{jit_suffix()}")
  end

  defp mlx_jit_enabled?, do: System.get_env("EMILY_MLX_JIT") == "1"

  defp jit_suffix, do: if(mlx_jit_enabled?(), do: "-jit", else: "")

  defp build_mlx(args) do
    dir = mlx_install_dir()

    if "--force" in args do
      File.rm_rf!(dir)
    end

    if File.dir?(dir) do
      {:ok, []}
    else
      do_build_mlx(dir)
      {:ok, []}
    end
  end

  defp do_build_mlx(install_dir) do
    File.mkdir_p!(cache_dir())
    build_dir = install_dir <> "-build"
    File.rm_rf!(build_dir)
    File.mkdir_p!(build_dir)

    ncpu =
      case :os.type() do
        {:unix, :darwin} ->
          {n, 0} = System.cmd("sysctl", ["-n", "hw.ncpu"])
          String.trim(n)

        _ ->
          {n, 0} = System.cmd("nproc", [])
          String.trim(n)
      end

    cmake_args = [
      "-S",
      @mlx_source_dir,
      "-B",
      build_dir,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_INSTALL_PREFIX=#{install_dir}",
      "-DBUILD_SHARED_LIBS=OFF",
      "-DMLX_BUILD_TESTS=OFF",
      "-DMLX_BUILD_EXAMPLES=OFF",
      "-DMLX_BUILD_BENCHMARKS=OFF",
      "-DMLX_BUILD_PYTHON_BINDINGS=OFF",
      "-DMLX_BUILD_SAFETENSORS=OFF",
      "-DMLX_BUILD_GGUF=OFF",
      "-DMLX_BUILD_METAL_TESTS=OFF",
      "-DMLX_METAL_JIT=#{if mlx_jit_enabled?(), do: "ON", else: "OFF"}"
    ]

    # When xcode-select points at CommandLineTools (the default on fresh
    # macOS), xcrun cannot find the Metal toolchain. If Xcode.app is
    # installed we set DEVELOPER_DIR so cmake and its custom commands
    # (make_compiled_preamble.sh) can resolve `xcrun -sdk macosx metal`.
    build_env = developer_dir_env()

    Mix.shell().info("Building MLX from source (#{mlx_cache_key()})...")

    maybe_apply_mlx_patches()

    run!("cmake", cmake_args, "cmake configure", build_env)
    run!("cmake", ["--build", build_dir, "--parallel", ncpu], "cmake build", build_env)
    run!("cmake", ["--install", build_dir], "cmake install", build_env)

    # Clean up the build directory — only the install prefix is needed.
    File.rm_rf!(build_dir)
    :ok
  end

  # Local patches applied to vendor/mlx before cmake runs. Each file
  # under patches/ is a unified diff rooted at the MLX source tree
  # (i.e. apply with `git -C vendor/mlx apply`). We apply idempotently
  # by checking whether the reverse patch applies cleanly first.
  @mlx_patches_dir Path.expand("patches", __DIR__)

  defp maybe_apply_mlx_patches do
    case File.ls(@mlx_patches_dir) do
      {:ok, entries} ->
        entries
        |> Enum.filter(&String.ends_with?(&1, ".patch"))
        |> Enum.sort()
        |> Enum.each(&apply_mlx_patch(Path.join(@mlx_patches_dir, &1)))

      _ ->
        :ok
    end
  end

  defp apply_mlx_patch(path) do
    # `-R --check` tests whether the patch is already applied. If so,
    # skip. Otherwise apply it.
    case System.cmd("git", ["-C", @mlx_source_dir, "apply", "-R", "--check", path],
           stderr_to_stdout: true
         ) do
      {_, 0} ->
        :already_applied

      _ ->
        case System.cmd("git", ["-C", @mlx_source_dir, "apply", path], stderr_to_stdout: true) do
          {_, 0} ->
            Mix.shell().info("  Applied MLX patch: #{Path.basename(path)}")
            :applied

          {output, code} ->
            Mix.raise("""
            Failed to apply MLX patch #{path} (exit #{code}):
            #{output}
            """)
        end
    end
  end

  @xcode_developer_dir "/Applications/Xcode.app/Contents/Developer"

  defp developer_dir_env do
    case System.cmd("xcrun", ["-sdk", "macosx", "metal", "--version"], stderr_to_stdout: true) do
      {_, 0} ->
        # Metal toolchain reachable from the default developer directory.
        []

      _ ->
        if File.dir?(@xcode_developer_dir) do
          Mix.shell().info("  Using Xcode.app for Metal toolchain (DEVELOPER_DIR)")
          [{"DEVELOPER_DIR", @xcode_developer_dir}]
        else
          Mix.raise("""
          Metal toolchain not found. MLX requires the Metal compiler.

          Install Xcode from the App Store and run:
            sudo xcode-select -s #{@xcode_developer_dir}

          Or, if Xcode is installed but the Metal Toolchain component is
          missing, run:
            xcodebuild -downloadComponent MetalToolchain
          """)
        end
    end
  end

  defp run!(cmd, args, label, env) do
    case System.cmd(cmd, args, stderr_to_stdout: true, env: env) do
      {_output, 0} ->
        :ok

      {output, code} ->
        Mix.raise("""
        #{label} failed (exit #{code}):
        #{output}
        """)
    end
  end
end
