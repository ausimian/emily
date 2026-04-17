defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.1.0"
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
      # Bumblebee >= 0.6.3 (the latest Hex release) lacks Qwen3 support.
      # Pinned to a `main` commit that contains `Bumblebee.Text.Qwen3` so
      # M4 can exercise Qwen3-0.6B end-to-end. Bump deliberately when a
      # newer release lands on Hex.
      {:bumblebee,
       github: "elixir-nx/bumblebee", ref: "273805e95507dc7866b958d90e0012a3abad1761", only: :test},
      {:tokenizers, "~> 0.5", only: :test},
      # Axon is already pulled in transitively by Bumblebee, but the M5
      # exit-criterion test (Axon MLP forward under `Emily.Compiler`)
      # reaches for it directly — pin it explicitly so the test isn't
      # hostage to a Bumblebee dep change.
      {:axon, "~> 0.7", only: :test},
      # `scidata` loads MNIST / CIFAR / etc. for the `:training_full`
      # opt-in convergence canary (M9). Kept test-only — Emily itself
      # doesn't depend on dataset loading.
      {:scidata, "~> 0.1", only: :test},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.34", only: :docs, runtime: false}
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
      source_url_pattern: "#{@source_url}/blob/v#{@version}/%{path}#L%{line}",
      extras: ["README.md", "PLAN.md", "CHANGELOG.md"]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files:
        ~w(lib c_src vendor/mlx/mlx vendor/mlx/cmake vendor/mlx/CMakeLists.txt vendor/mlx/LICENSE Makefile mix.exs README.md CHANGELOG.md LICENSE)
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
    Path.join(cache_dir(), "mlx-#{mlx_cache_key()}")
  end

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
      "-DUSE_SYSTEM_FMT=ON"
    ]

    # When xcode-select points at CommandLineTools (the default on fresh
    # macOS), xcrun cannot find the Metal toolchain. If Xcode.app is
    # installed we set DEVELOPER_DIR so cmake and its custom commands
    # (make_compiled_preamble.sh) can resolve `xcrun -sdk macosx metal`.
    build_env = developer_dir_env()

    Mix.shell().info("Building MLX from source (#{mlx_cache_key()})...")

    run!("cmake", cmake_args, "cmake configure", build_env)
    run!("cmake", ["--build", build_dir, "--parallel", ncpu], "cmake build", build_env)
    run!("cmake", ["--install", build_dir], "cmake install", build_env)

    # Clean up the build directory — only the install prefix is needed.
    File.rm_rf!(build_dir)
    :ok
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
