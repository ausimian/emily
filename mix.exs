defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.1.0"
  @mlx_version "0.25.1"
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
      extra_applications: [:logger, :inets, :ssl, :public_key, :crypto],
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
      "compile.emily_mlx": &fetch_mlx/1
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
      files: ~w(lib c_src Makefile mix.exs README.md CHANGELOG.md LICENSE)
    ]
  end

  # ---------- MLX fetching ----------

  @target_triples %{
    {:unix, :darwin, "aarch64"} => "arm64-apple-darwin",
    {:unix, :darwin, "arm64"} => "arm64-apple-darwin",
    {:unix, :darwin, "x86_64"} => "x86_64-apple-darwin"
  }

  defp make_env do
    dir = mlx_dir()

    %{
      "MLX_DIR" => dir,
      "MLX_INCLUDE_DIR" => Path.join(dir, "include"),
      "MLX_LIB_DIR" => Path.join(dir, "lib"),
      "MLX_VERSION" => @mlx_version,
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

  defp mlx_dir do
    Path.join(cache_dir(), "mlx-#{@mlx_version}-#{current_target!()}")
  end

  defp current_target! do
    {family, name} = :os.type()
    arch = :erlang.system_info(:system_architecture) |> to_string() |> arch_prefix()

    case Map.fetch(@target_triples, {family, name, arch}) do
      {:ok, triple} ->
        triple

      :error ->
        Mix.raise("""
        emily currently supports macOS (arm64 or x86_64) only.
        Detected: family=#{family} name=#{name} arch=#{arch}
        """)
    end
  end

  defp arch_prefix(sysarch) do
    sysarch |> String.split("-", parts: 2) |> hd()
  end

  defp fetch_mlx(args) do
    dir = mlx_dir()

    if "--force" in args do
      File.rm_rf!(dir)
    end

    cond do
      File.dir?(dir) ->
        {:ok, []}

      true ->
        do_fetch_mlx(dir)
        {:ok, []}
    end
  end

  defp do_fetch_mlx(dir) do
    File.mkdir_p!(cache_dir())

    target = current_target!()
    archive = Path.join(cache_dir(), "mlx-#{@mlx_version}-#{target}.tar.gz")

    unless File.exists?(archive) do
      url =
        "https://github.com/cocoa-xu/mlx-build/releases/download/" <>
          "v#{@mlx_version}/mlx-#{target}.tar.gz"

      Mix.shell().info("Fetching MLX #{@mlx_version} for #{target}")
      Mix.shell().info("  from #{url}")
      download!(url, archive)

      sha_url = url <> ".sha256"
      verify_sha256!(archive, sha_url)
    end

    parent = Path.join(Path.dirname(dir), "mlx-extract-#{@mlx_version}")
    File.rm_rf!(parent)
    File.mkdir_p!(parent)

    :ok = :erl_tar.extract(archive, [:compressed, {:cwd, String.to_charlist(parent)}])
    File.rename!(parent, dir)
    :ok
  end

  defp verify_sha256!(archive, sha_url) do
    expected =
      sha_url
      |> download!()
      |> String.split(" ", parts: 2, trim: true)
      |> hd()

    actual =
      archive
      |> File.read!()
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)

    if actual != expected do
      File.rm_rf!(archive)

      Mix.raise("""
      MLX archive checksum mismatch.
        archive:  #{archive}
        expected: #{expected}
        actual:   #{actual}
      """)
    end

    :ok
  end

  defp download!(url, save_as \\ nil) do
    :inets.start()
    :ssl.start()

    https_opts = [
      ssl:
        [
          verify: :verify_peer,
          depth: 5,
          customize_hostname_check: [
            match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
          ]
        ] ++ cacerts_options()
    ]

    request = {String.to_charlist(url), []}

    case :httpc.request(:get, request, https_opts, body_format: :binary) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        if save_as, do: File.write!(save_as, body)
        body

      {:ok, {{_, code, _}, _headers, _body}} ->
        Mix.raise("HTTP #{code} fetching #{url}")

      other ->
        Mix.raise("Failed to fetch #{url}: #{inspect(other)}")
    end
  end

  defp cacerts_options do
    try do
      [cacerts: :public_key.cacerts_get()]
    rescue
      _ -> []
    end
  end
end
