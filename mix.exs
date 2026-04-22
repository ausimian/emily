defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.2.1"
  @source_url "https://github.com/ausimian/emily"

  # MLX pin. The checksums map keys every supported (os, arch, variant)
  # asset hosted on the `mlx-#{@mlx_version}` release of this repo and
  # is verified against every downloaded tarball before extraction. Bumps
  # must ship the new SHA256s in the same commit — see the release-mlx
  # workflow for how to cut new prebuilts.
  @mlx_version "0.31.2"
  @mlx_checksums %{
    "mlx-0.31.2-macos-arm64-aot.tar.gz" =>
      "d752d33ea9ef050541263c97c87a47cbc72a239f9a2e355ae73b941bf24be012",
    "mlx-0.31.2-macos-arm64-jit.tar.gz" =>
      "8982b126697ed422c4b5a17e8171a3bbcf661383fdac8f01ac9ddf5cc309d9e3"
  }

  # Read once at project load — `:mlx_variant` lives in `config/config.exs`
  # so CI can flip it via `config/local.exs` without a custom MIX_ENV.
  # The env key is needed because config/config.exs calls `config_env/0`
  # to pick a sibling env file.
  @emily_cfg Path.expand("config/config.exs", __DIR__)
             |> Config.Reader.read!(env: Mix.env())
             |> Keyword.fetch!(:emily)

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
      files: ~w(lib c_src Makefile mix.exs config README.md CHANGELOG.md LICENSE)
    ]
  end

  # ---------- MLX prebuilt download ----------

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
    case Keyword.fetch!(@emily_cfg, :mlx_variant) do
      :no_jit ->
        "aot"

      :jit ->
        "jit"

      other ->
        Mix.raise("""
        Invalid :mlx_variant #{inspect(other)} in config/config.exs.
        Expected :no_jit or :jit.
        """)
    end
  end

  defp mlx_install_dir,
    do: Path.join(cache_dir(), "mlx-#{@mlx_version}-#{mlx_variant()}")

  defp mlx_asset_name, do: "mlx-#{@mlx_version}-macos-#{arch_tag()}-#{mlx_variant()}.tar.gz"

  defp arch_tag do
    case {:os.type(), :erlang.system_info(:system_architecture) |> to_string()} do
      {{:unix, :darwin}, "aarch64" <> _} ->
        "arm64"

      {{:unix, :darwin}, "x86_64" <> _} ->
        Mix.raise("""
        No x86_64 macOS prebuilt exists for MLX #{@mlx_version}.
        Apple Silicon is required; x86_64 Macs aren't supported by this build.
        """)

      {os, arch} ->
        Mix.raise("""
        Emily's MLX prebuilts are macOS-only; cannot build on
        #{inspect(os)} / #{arch}.
        """)
    end
  end

  defp build_mlx(args) do
    dir = mlx_install_dir()

    if "--force" in args do
      File.rm_rf!(dir)
    end

    if File.dir?(dir) do
      {:ok, []}
    else
      download_and_extract_mlx(dir)
      {:ok, []}
    end
  end

  defp download_and_extract_mlx(install_dir) do
    asset = mlx_asset_name()
    url = "https://github.com/ausimian/emily/releases/download/mlx-#{@mlx_version}/#{asset}"
    tarball = Path.join(cache_dir(), asset)

    expected =
      Map.get(@mlx_checksums, asset) ||
        Mix.raise("""
        No SHA256 pinned for #{asset} in @mlx_checksums (mix.exs).
        This usually means the pinned version was bumped without
        updating the checksum map.
        """)

    File.mkdir_p!(cache_dir())
    File.mkdir_p!(install_dir)

    Mix.shell().info("Downloading MLX prebuilt #{asset}")
    http_download!(url, tarball)
    verify_sha256!(tarball, expected)

    case System.cmd(
           "tar",
           ["-xzf", tarball, "-C", install_dir, "--strip-components=1"],
           stderr_to_stdout: true
         ) do
      {_, 0} ->
        :ok

      {output, code} ->
        Mix.raise("""
        tar extract failed (exit #{code}):
        #{output}
        """)
    end

    File.rm(tarball)
    :ok
  end

  defp http_download!(url, dest) do
    {:ok, _} = Application.ensure_all_started(:inets)
    {:ok, _} = Application.ensure_all_started(:ssl)
    # `ensure_all_started(:ssl)` transitively starts the :public_key app,
    # but a consumer's `mix compile` sometimes reaches this point before
    # the :public_key module itself has been loaded — smoke-test CI on a
    # clean cache hit "(UndefinedFunctionError) :public_key.cacerts_get/0
    # ... module :public_key is not available". Force a module load.
    {:module, :public_key} = :code.ensure_loaded(:public_key)

    http_opts = [
      autoredirect: true,
      ssl: [
        verify: :verify_peer,
        cacerts: :public_key.cacerts_get(),
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
      ]
    ]

    request = {String.to_charlist(url), []}
    opts = [body_format: :binary, stream: String.to_charlist(dest)]

    case :httpc.request(:get, request, http_opts, opts) do
      {:ok, :saved_to_file} ->
        :ok

      {:ok, {{_, 200, _}, _headers, _body}} ->
        :ok

      {:ok, {{_, status, reason}, _headers, _body}} ->
        File.rm(dest)
        Mix.raise("MLX download failed (HTTP #{status} #{reason}): #{url}")

      {:error, reason} ->
        File.rm(dest)
        Mix.raise("MLX download failed (#{inspect(reason)}): #{url}")
    end
  end

  defp verify_sha256!(path, expected) do
    actual =
      path
      |> File.read!()
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)

    if actual != expected do
      File.rm(path)

      Mix.raise("""
      MLX prebuilt checksum mismatch for #{Path.basename(path)}.
        expected: #{expected}
        actual:   #{actual}
      """)
    end
  end
end
