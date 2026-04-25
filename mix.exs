defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.3.1"
  @source_url "https://github.com/ausimian/emily"

  # MLX pin. Drives the git tag the `:mlx_src` dep is cloned at (see
  # `deps/0`) and the per-variant cache dir layout. Bump in lockstep with
  # the submodule ref; CI's `release-nif.yml` rebuilds the NIF against
  # whatever this resolves to.
  @mlx_version "0.31.2"

  # Precompiled NIF targets this `@version` ships. Used as an
  # early fail-fast guard in the hex-consumer fetch step (an
  # unsupported target 404s on the tarball download, but the list
  # lets us raise with a clearer message). Checksums are not pinned
  # here; the consumer fetches a `.sha256` sidecar alongside each
  # tarball at compile time.
  @supported_targets [
    {:aot, "macos-arm64"},
    {:jit, "macos-arm64"}
  ]

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
      compilers: compilers(),
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

  # Dev / CI checkout has `c_src/` on disk — compile the NIF from source
  # against a locally-built libmlx. Hex consumers don't get `c_src/` in
  # their tarball (see `package[:files]`), so we download a precompiled
  # NIF tarball instead. Detection by filesystem is the simplest thing
  # that doesn't need a new env var or runtime flag.
  defp compilers do
    if File.dir?("c_src") do
      [:emily_mlx, :elixir_make] ++ Mix.compilers()
    else
      [:emily_nif] ++ Mix.compilers()
    end
  end

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
      "compile.emily_mlx": &build_mlx/1,
      "compile.emily_nif": &fetch_nif/1,
      # The MLX install dir lives in the user-level cache and is
      # *deliberately* preserved across `mix clean` (rebuilding from
      # source is ~5–7 min). Wipe it explicitly with `mix clean.mlx`.
      "clean.mlx": &clean_mlx/1
    ]
  end

  defp clean_mlx(_args) do
    case Path.wildcard(Path.join(cache_dir(), "mlx-*")) do
      [] ->
        Mix.shell().info("No MLX install dirs to clean in #{cache_dir()}")

      dirs ->
        for dir <- dirs do
          File.rm_rf!(dir)
          Mix.shell().info("Removed #{dir}")
        end
    end
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
      files: ~w(lib mix.exs README.md CHANGELOG.md LICENSE)
    ]
  end

  # ---------- MLX source build (in-repo / CI) ----------

  defp make_env do
    dir = mlx_install_dir()

    %{
      "MLX_DIR" => dir,
      "MLX_INCLUDE_DIR" => Path.join(dir, "include"),
      "MLX_LIB_DIR" => Path.join(dir, "lib"),
      "FINE_INCLUDE_DIR" => Fine.include_dir()
    }
  end

  # Default cache location, override with `EMILY_CACHE`. On macOS we
  # use `DARWIN_USER_CACHE_DIR` (`/private/var/folders/<hash>/C/emily`)
  # — the per-user sandboxed cache root that Apple's own sandboxed apps
  # use for transient-but-persistent state. It's per-user, persistent
  # across reboots (unlike `/tmp`), and lives outside `~/Library/` so
  # it's not subject to user-facing backup/sync tooling defaults.
  # Linux / Windows fall back to the XDG convention.
  defp cache_dir do
    case System.get_env("EMILY_CACHE") do
      nil -> default_cache_dir()
      dir -> Path.expand(dir)
    end
  end

  defp default_cache_dir do
    case :os.type() do
      {:unix, :darwin} ->
        {out, 0} = System.cmd("getconf", ["DARWIN_USER_CACHE_DIR"])
        Path.join(String.trim(out), "emily")

      _ ->
        cache_home = System.get_env("XDG_CACHE_HOME") || Path.join(System.user_home!(), ".cache")
        Path.join(cache_home, "emily")
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

    if mlx_installed?(dir) do
      {:ok, []}
    else
      File.rm_rf!(dir)
      build_mlx_from_source!(dir)
      {:ok, []}
    end
  end

  # `build-mlx.sh` publishes via `mv staging prefix` only after the
  # critical artefacts are in place, so a directory missing either of
  # these is by definition a partial install — don't trust it.
  defp mlx_installed?(dir) do
    File.exists?(Path.join([dir, "lib", "libmlx.a"])) and
      File.exists?(Path.join([dir, "lib", "mlx.metallib"]))
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

  # ---------- Precompiled NIF download (hex consumer) ----------

  defp fetch_nif(_args) do
    variant = Application.get_env(:emily, :variant, :aot)
    target = detect_nif_target!()
    key = {variant, target}

    unless key in @supported_targets do
      Mix.raise("""
      No precompiled NIF for #{inspect(key)} on emily #{@version}.
      Supported: #{inspect(@supported_targets)}.
      """)
    end

    asset = "emily-nif-#{@version}-#{variant}-#{target}.tar.gz"
    base_url = "#{@source_url}/releases/download/#{@version}"

    cache = cache_dir()
    File.mkdir_p!(cache)
    tarball = Path.join(cache, asset)
    sha_path = tarball <> ".sha256"

    priv = Path.join(Mix.Project.app_path(), "priv")
    File.mkdir_p!(priv)

    # Fetch the sidecar on every compile — it's a tiny file and
    # drives verification of the (much larger) tarball. Lets us
    # re-verify a cached tarball against whatever's currently
    # published, instead of trusting the local disk copy blind.
    http_download!("#{base_url}/#{asset}.sha256", sha_path)
    expected = sha_path |> File.read!() |> String.split() |> hd()

    unless File.exists?(tarball) and sha256_ok?(tarball, expected) do
      Mix.shell().info("Downloading precompiled NIF #{asset}")
      http_download!("#{base_url}/#{asset}", tarball)
      verify_sha256!(tarball, expected)
    end

    case System.cmd("tar", ["-xzf", tarball, "-C", priv], stderr_to_stdout: true) do
      {_, 0} ->
        :ok

      {output, code} ->
        Mix.raise("""
        tar extract failed (exit #{code}):
        #{output}
        """)
    end

    {:ok, []}
  end

  defp detect_nif_target! do
    case {:os.type(), :erlang.system_info(:system_architecture) |> to_string()} do
      {{:unix, :darwin}, "aarch64" <> _} ->
        "macos-arm64"

      {{:unix, :darwin}, "x86_64" <> _} ->
        Mix.raise("""
        No precompiled NIF for x86_64 macOS — Emily is Apple Silicon only.
        """)

      {os, arch} ->
        Mix.raise("""
        No precompiled NIF for #{inspect(os)} / #{arch}.
        Supported targets: #{inspect(Enum.uniq(Enum.map(@supported_targets, &elem(&1, 1))))}.
        """)
    end
  end

  # Mix prunes the parent VM's code path during dep compilation: the
  # ebin dirs for :ssl, :public_key, :asn1 and most of :inets become
  # unreachable even though their apps report as loaded/started. Run
  # the whole HTTPS round-trip on a peer node instead — the child VM
  # spawned by :peer has a fresh, un-pruned code path, so standard
  # httpc + public_key just work.
  defp http_download!(url, dest) do
    {:ok, pid, _node} =
      :peer.start_link(%{connection: :standard_io, name: :peer.random_name()})

    try do
      {:ok, _} = :peer.call(pid, :application, :ensure_all_started, [:inets])
      {:ok, _} = :peer.call(pid, :application, :ensure_all_started, [:ssl])

      cacerts = :peer.call(pid, :public_key, :cacerts_get, [])
      match_fun = :peer.call(pid, :public_key, :pkix_verify_hostname_match_fun, [:https])

      http_opts = [
        autoredirect: true,
        ssl: [
          verify: :verify_peer,
          cacerts: cacerts,
          customize_hostname_check: [match_fun: match_fun]
        ]
      ]

      request = {String.to_charlist(url), []}
      opts = [body_format: :binary, stream: String.to_charlist(dest)]

      # :peer.call/4 defaults to a 5000 ms gen_server.call timeout. The
      # sha256 sidecar fits; a multi-MB tarball doesn't. Let httpc drive
      # its own timing and don't let the RPC wrapper abort it.
      case :peer.call(pid, :httpc, :request, [:get, request, http_opts, opts], :infinity) do
        {:ok, :saved_to_file} ->
          :ok

        {:ok, {{_, 200, _}, _headers, _body}} ->
          :ok

        {:ok, {{_, status, reason}, _headers, _body}} ->
          File.rm(dest)
          Mix.raise("NIF download failed (HTTP #{status} #{reason}): #{url}")

        {:error, reason} ->
          File.rm(dest)
          Mix.raise("NIF download failed (#{inspect(reason)}): #{url}")
      end
    after
      :peer.stop(pid)
    end
  end

  defp sha256_ok?(path, expected) do
    path
    |> File.read!()
    |> then(&:crypto.hash(:sha256, &1))
    |> Base.encode16(case: :lower) == expected
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
      NIF tarball checksum mismatch for #{Path.basename(path)}.
        expected: #{expected}
        actual:   #{actual}
      """)
    end
  end
end
