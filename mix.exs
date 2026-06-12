defmodule Emily.MixProject do
  use Mix.Project

  @app :emily
  @version "0.6.1"
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
    [
      preferred_envs: [
        docs: :docs,
        "hex.publish": :docs,
        "emily.publish": :docs,
        precommit: :test
      ]
    ]
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
      {:nx, "~> 0.12"},
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
      {:bumblebee, "~> 0.7", optional: true},
      {:tokenizers, "~> 0.5", optional: true},
      {:axon, "~> 0.8", optional: true},
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
        &docs_check/1,
        "test"
      ],
      "compile.emily_mlx": &build_mlx/1,
      "compile.emily_nif": &fetch_nif/1,
      # The MLX install dir lives in the user-level cache and is
      # *deliberately* preserved across `mix clean` (rebuilding from
      # source is ~5–7 min). Wipe it explicitly with `mix clean.mlx`.
      "clean.mlx": &clean_mlx/1,
      # `mix emily.publish` regenerates the pinned NIF checksums
      # (`native_checksums.txt`, git-ignored) from the freshly-built release
      # artifacts, leaving nothing to commit by hand. Publishing is a
      # deliberate *second* step — `mix hex.publish` — not chained here: Mix
      # only loads the Hex archive for the task named on the command line, so
      # a `hex.publish` step inside this alias fails with "task could not be
      # found". `emily.checksums` downloads the published artifacts, so it
      # refuses to run until the release assets are public. See MAINTAINING.md.
      "emily.publish": ["emily.checksums"]
    ]
  end

  # Build the *published* doc surface (the `:docs` env that `mix hex.publish`
  # uses) and fail on any ExDoc warning — autolinks to hidden/undefined
  # symbols, broken refs, etc. Run as a subprocess because `mix precommit`
  # itself runs in `:test`, where `test/support/` modules (and their
  # `only: :test` deps like Axon) compile in and would warn for code that
  # never ships. The NIF in `priv/` is shared across envs, so this reuses it
  # — no native rebuild, just an Elixir recompile + doc generation. CI runs
  # `mix precommit`, so this gates merges too.
  defp docs_check(_args) do
    {output, status} =
      System.cmd("mix", ["docs", "--warnings-as-errors"],
        env: [{"MIX_ENV", "docs"}],
        stderr_to_stdout: true
      )

    IO.write(output)

    if status != 0 do
      Mix.raise(
        "mix docs reported warnings (see above). Fix the reference, or if it " <>
          "points at a hidden/undefined symbol, add it to :skip_code_autolink_to in docs/0."
      )
    end
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
      # Symbols ExDoc can't link, so it warns on every reference to them.
      # Listed explicitly on purpose: the `mix precommit` docs gate fails
      # with the exact unlinkable symbol when a new one appears, making each
      # addition here a deliberate one-liner rather than silent auto-skipping.
      # A module entry (`Emily.Native`) does NOT cover its members, so each
      # referenced function is listed too.
      skip_code_autolink_to: [
        # `Emily.Native` and its NIF shims are `@moduledoc false`.
        "Emily.Native",
        "Emily.Native.from_binary/3",
        "Emily.Native.conv_general/8",
        "Emily.Native.worker_queue_depth/1",
        # Native Expr-compiler internals are `@moduledoc false`.
        "Emily.IR",
        "Emily.Program",
        # Hidden Nx callback + private/external Nx internals.
        "Emily.Backend.block/4",
        "Nx.Backend.block/4",
        "Nx.Defn.Expr.optional/3"
      ],
      extras: [
        "README.md",
        "ARCHITECTURE.md",
        "ROADMAP.md",
        "CHANGELOG.md",
        "livebooks/distilbert_qa.livemd",
        "livebooks/qwen3_quantized.livemd",
        "livebooks/nomic_embeddings.livemd",
        "livebooks/smollm3_chat.livemd",
        "livebooks/modernbert_classification.livemd",
        "livebooks/mnist_training.livemd",
        "livebooks/whisper_transcription.livemd",
        "livebooks/fast_kernels.livemd"
      ],
      groups_for_extras: [
        README: ~r{README.md},
        Project: [
          "ARCHITECTURE.md",
          "ROADMAP.md",
          "CHANGELOG.md"
        ],
        Livebooks: ~r{^livebooks/}
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
        Observability: [Emily.Telemetry, Emily.Memory]
      ]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files:
        ~w(lib mix.exs native_checksums.txt README.md ARCHITECTURE.md ROADMAP.md CHANGELOG.md LICENSE)
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
        {out, 0} = System.cmd("/usr/bin/getconf", ["DARWIN_USER_CACHE_DIR"])
        Path.join(String.trim(out), "emily")

      _ ->
        cache_home = System.get_env("XDG_CACHE_HOME") || Path.join(System.user_home!(), ".cache")
        Path.join(cache_home, "emily")
    end
  end

  # ---------- Cache-dir trust (dev/CI source build) ----------

  # The MLX install dir is statically linked into libemily, so a planted
  # libmlx.a is arbitrary native code in the BEAM. Refuse to trust (or
  # reuse) a cache/install dir owned by another user — the exposure is a
  # shared EMILY_CACHE on a multi-user host — and keep our own dirs 0700.
  defp current_uid do
    {out, 0} = System.cmd("/usr/bin/id", ["-u"])
    out |> String.trim() |> String.to_integer()
  end

  defp assert_owned!(dir) do
    case File.stat(dir) do
      {:ok, %File.Stat{uid: uid}} ->
        unless uid == current_uid() do
          Mix.raise("""
          Refusing to trust #{dir}: it is owned by uid #{uid}, not you
          (uid #{current_uid()}). A shared or attacker-controlled cache
          could plant a malicious libmlx.a that is statically linked into
          the NIF. Point EMILY_CACHE at a private, user-owned directory.
          """)
        end

      {:error, :enoent} ->
        :ok

      {:error, reason} ->
        Mix.raise("Cannot stat #{dir}: #{inspect(reason)}")
    end
  end

  defp prepare_cache_dir! do
    cache = cache_dir()
    File.mkdir_p!(cache)
    assert_owned!(cache)
    File.chmod!(cache, 0o700)
    cache
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
    assert_owned!(cache_dir())
    dir = mlx_install_dir()
    assert_owned!(dir)

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

    prepare_cache_dir!()

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

    cache = cache_dir()
    File.mkdir_p!(cache)
    tarball = Path.join(cache, asset)

    priv = Path.join(Mix.Project.app_path(), "priv")
    File.mkdir_p!(priv)

    with_nif_artifact(fn ->
      # Verify against the checksum pinned in the hex package, not a
      # sidecar fetched from the same (mutable) release as the tarball.
      expected = pinned_checksum!(asset)

      unless File.exists?(tarball) and sha256_ok?(tarball, expected) do
        Mix.shell().info("Downloading precompiled NIF #{asset}")
        http_download!("#{@source_url}/releases/download/#{@version}/#{asset}", tarball)
        verify_sha256!(tarball, expected)
      end

      extract_nif!(tarball, priv)
    end)

    {:ok, []}
  end

  # Expected SHA-256 for `asset`, read from the in-package
  # `native_checksums.txt` (see `Emily.NifArtifact`) so trust is rooted in
  # the Hex-immutable package rather than the mutable GitHub release.
  defp pinned_checksum!(asset) do
    path = Path.expand("native_checksums.txt", __DIR__)

    unless File.exists?(path) do
      Mix.raise("""
      Missing native_checksums.txt — cannot verify the precompiled NIF.
      A maintainer must pin checksums with `mix emily.checksums`.
      """)
    end

    checksums = path |> File.read!() |> Emily.NifArtifact.parse_checksums()

    case Emily.NifArtifact.expected(checksums, asset) do
      {:ok, hex} ->
        hex

      :error ->
        Mix.raise("""
        No pinned checksum for #{asset} in native_checksums.txt.
        The file may predate emily #{@version}; regenerate it with
        `mix emily.checksums`.
        """)
    end
  end

  # Validate the tarball's entries against an allowlist, then extract only
  # the allowlisted regular files via :erl_tar (no shelling to a
  # $PATH-resolved `tar`). Guards against path-traversal, symlink, and
  # unexpected-entry attacks in a tarball pulled from a mutable release.
  defp extract_nif!(tarball, priv) do
    charlist = String.to_charlist(tarball)

    entries =
      case :erl_tar.table(charlist, [:compressed, :verbose]) do
        {:ok, table} ->
          Enum.map(table, fn entry -> {to_string(elem(entry, 0)), elem(entry, 1)} end)

        {:error, reason} ->
          Mix.raise("Could not read NIF tarball #{tarball}: #{inspect(reason)}")
      end

    case Emily.NifArtifact.verify_entries(entries) do
      :ok -> :ok
      {:error, msg} -> Mix.raise("Refusing to extract NIF tarball: #{msg}")
    end

    files = Enum.map(Emily.NifArtifact.allowlist(), &String.to_charlist/1)
    opts = [:compressed, {:cwd, String.to_charlist(priv)}, {:files, files}]

    case :erl_tar.extract(charlist, opts) do
      :ok -> :ok
      {:error, reason} -> Mix.raise("NIF extract failed: #{inspect(reason)}")
    end
  end

  # Load the pure helper module (`lib/` is not yet compiled at the
  # `:emily_nif` compiler stage), then purge it (if we loaded it) so the
  # regular elixir compiler can load its own copy without a
  # "redefining module" warning.
  defp with_nif_artifact(fun) do
    preloaded? = Code.ensure_loaded?(Emily.NifArtifact)
    unless preloaded?, do: Code.require_file("lib/emily/nif_artifact.ex", __DIR__)

    try do
      fun.()
    after
      unless preloaded? do
        :code.purge(Emily.NifArtifact)
        :code.delete(Emily.NifArtifact)
      end
    end
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
