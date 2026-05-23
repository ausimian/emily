defmodule Mix.Tasks.Emily.Doctor do
  @moduledoc """
  Diagnose the local Emily runtime installation.

  The task checks the host platform, active MLX variant, required
  `priv/` artifacts, NIF loadability, and a tiny Emily backend smoke
  test. Checks short-circuit: when a prerequisite fails, dependent
  checks are reported as `[skip]` instead of running and producing
  cascading noise.

  ## Flags

      --help        Print this help text and exit.
      --variant     Override the active variant (`aot` or `jit`) for
                    this run, e.g. `--variant jit`. Does not mutate
                    config — used to ask "would this host satisfy
                    :jit?".
  """

  use Mix.Task

  @requirements ["app.config"]
  @shortdoc "Diagnose the local Emily runtime installation"

  @aot_min_macos {14, 0, 0}
  @jit_min_macos {26, 2, 0}

  @switches [help: :boolean, variant: :string]

  @type status :: :ok | :warn | :error | :skip
  @type check :: %{
          name: String.t(),
          status: status(),
          summary: String.t(),
          guidance: [String.t()]
        }

  @impl Mix.Task
  def run(args) do
    case OptionParser.parse(args, strict: @switches) do
      {opts, [], []} -> dispatch(opts)
      {_, positional, _} when positional != [] -> raise_positional(positional)
      {_, _, invalid} -> raise_invalid_flags(invalid)
    end
  end

  defp dispatch(opts) do
    if opts[:help] do
      Mix.shell().info(@moduledoc)
    else
      report = diagnose(runtime_opts(opts))
      Mix.shell().info(format_report(report))

      if report.status == :error do
        Mix.raise("emily.doctor found #{count(report, :error)} problem(s)")
      end
    end
  end

  defp raise_positional(positional) do
    Mix.raise("emily.doctor takes no positional arguments, got: #{inspect(positional)}")
  end

  defp raise_invalid_flags(invalid) do
    flags = Enum.map_join(invalid, ", ", fn {f, _} -> f end)
    Mix.raise("emily.doctor: unknown flag(s): #{flags}. Run `mix emily.doctor --help`.")
  end

  defp runtime_opts(opts) do
    case opts[:variant] do
      nil -> []
      "aot" -> [variant: :aot]
      "jit" -> [variant: :jit]
      other -> Mix.raise("emily.doctor: --variant must be aot or jit, got: #{inspect(other)}")
    end
  end

  @doc false
  @spec diagnose(keyword()) :: %{status: :ok | :warn | :error, checks: [check()]}
  def diagnose(opts \\ []) do
    context = %{
      os_type: Keyword.get_lazy(opts, :os_type, &:os.type/0),
      arch: Keyword.get_lazy(opts, :arch, fn -> :erlang.system_info(:system_architecture) end),
      macos_version: Keyword.get_lazy(opts, :macos_version, &macos_version/0),
      variant:
        Keyword.get_lazy(opts, :variant, fn -> Application.get_env(:emily, :variant, :aot) end),
      priv_dir: Keyword.get_lazy(opts, :priv_dir, &priv_dir/0),
      file_exists?: Keyword.get(opts, :file_exists?, &File.exists?/1),
      nif_loader: Keyword.get(opts, :nif_loader, &load_nif/0),
      smoke_test: Keyword.get(opts, :smoke_test, &smoke_test/0)
    }

    platform = platform_check(context)
    variant = variant_check(context)

    nif_artifact = gate([platform], "priv/libemily", fn -> nif_artifact_check(context) end)
    metallib = gate([platform], "priv/mlx.metallib", fn -> metallib_artifact_check(context) end)

    nif_load =
      gate([platform, nif_artifact], "NIF loadability", fn -> nif_load_check(context) end)

    smoke =
      gate(
        [platform, variant, nif_artifact, metallib, nif_load],
        "smoke test",
        fn -> smoke_check(context) end
      )

    checks = [platform, variant, nif_artifact, metallib, nif_load, smoke]
    %{status: rollup(checks), checks: checks}
  end

  defp rollup(checks) do
    cond do
      Enum.any?(checks, &(&1.status == :error)) -> :error
      Enum.any?(checks, &(&1.status == :warn)) -> :warn
      true -> :ok
    end
  end

  defp gate(prereqs, name, fun) do
    case Enum.find(prereqs, &(&1.status == :error)) do
      nil -> fun.()
      failed -> skip(name, "skipped (#{failed.name} #{failed.status})")
    end
  end

  @doc false
  @spec format_report(%{status: :ok | :warn | :error, checks: [check()]}) :: String.t()
  def format_report(report) do
    header =
      case report.status do
        :ok -> "Emily doctor: OK"
        :warn -> "Emily doctor: #{count(report, :warn)} warning(s)"
        :error -> "Emily doctor: #{count(report, :error)} problem(s) found"
      end

    ([header, ""] ++ Enum.map(report.checks, &format_check/1))
    |> Enum.join("\n")
  end

  defp platform_check(%{os_type: os_type, arch: arch, macos_version: version, variant: variant}) do
    arch = to_string(arch)
    parsed_version = parse_macos_version(version)

    cond do
      os_type != {:unix, :darwin} ->
        error("platform", "unsupported OS #{inspect(os_type)}",
          guidance: [
            "Emily's MLX runtime is macOS-only.",
            "Run on an Apple Silicon Mac or use a non-Emily Nx backend on this host."
          ]
        )

      not apple_silicon?(arch) ->
        error("platform", "unsupported architecture #{arch}",
          guidance: [
            "Emily prebuilt NIFs target Apple Silicon arm64.",
            "x86_64 Macs, Rosetta shells, Linux, and Windows are not supported by the MLX runtime."
          ]
        )

      parsed_version == :unknown ->
        warn("platform", "macOS version could not be detected on #{arch}",
          guidance: [
            "Ensure :aot runs on macOS #{format_version(@aot_min_macos)} or newer.",
            "Ensure :jit runs on macOS #{format_version(@jit_min_macos)} or newer."
          ]
        )

      version_too_old?(parsed_version, variant) ->
        {required, label} = minimum_macos(variant)

        error("platform", "macOS #{format_version(parsed_version)} is too old for #{label}",
          guidance: [
            "#{label} requires macOS #{format_version(required)} or newer.",
            "Use :aot on older macOS hosts, or upgrade macOS before using :jit."
          ]
        )

      true ->
        ok("platform", "macOS #{format_version(parsed_version)} on #{arch}")
    end
  end

  defp variant_check(%{variant: variant}) when variant in [:aot, :jit] do
    ok("variant", "active #{inspect(variant)}")
  end

  defp variant_check(%{variant: variant}) do
    error("variant", "invalid :emily, :variant #{inspect(variant)}",
      guidance: [
        "Set `config :emily, variant: :aot` or `config :emily, variant: :jit`.",
        "For this repo's config, set EMILY_MLX_VARIANT=aot or EMILY_MLX_VARIANT=jit."
      ]
    )
  end

  defp nif_artifact_check(%{priv_dir: :error}), do: missing_priv("priv/libemily")

  defp nif_artifact_check(%{priv_dir: priv_dir, file_exists?: file_exists?}) do
    path = Path.join(priv_dir, "libemily.so")

    if file_exists?.(path) do
      ok("priv/libemily", path)
    else
      error("priv/libemily", "missing #{path}",
        guidance: [
          "Run `mix compile` to build or download Emily's NIF.",
          "If the build was interrupted, run `mix clean` and then `mix compile`."
        ]
      )
    end
  end

  defp metallib_artifact_check(%{priv_dir: :error}), do: missing_priv("priv/mlx.metallib")

  defp metallib_artifact_check(%{priv_dir: priv_dir, file_exists?: file_exists?}) do
    path = Path.join(priv_dir, "mlx.metallib")

    if file_exists?.(path) do
      ok("priv/mlx.metallib", path)
    else
      error("priv/mlx.metallib", "missing #{path}",
        guidance: [
          "`mlx.metallib` must be colocated with `libemily` for MLX Metal kernels.",
          "Run `mix compile`; if switching :aot/:jit variants, run `mix clean` first."
        ]
      )
    end
  end

  defp missing_priv(name) do
    error(name, "could not locate :emily priv directory",
      guidance: [
        "`:code.priv_dir(:emily)` failed — :emily may not be on the loadpath.",
        "Run `mix deps.compile emily` from the consumer project, or `mix compile` here."
      ]
    )
  end

  defp nif_load_check(%{nif_loader: loader}) do
    case safe_call(loader) do
      {:ok, summary} ->
        ok("NIF loadability", summary)

      {:error, reason} ->
        error("NIF loadability", "failed to load Emily.Native: #{reason}",
          guidance: [
            "Confirm `priv/libemily` matches this host, OTP, and active Emily variant.",
            "Run `mix compile --force` after changing variants or moving build artifacts."
          ]
        )
    end
  end

  defp smoke_check(%{smoke_test: smoke_test}) do
    case safe_call(smoke_test) do
      {:ok, summary} ->
        ok("smoke test", summary)

      {:error, reason} ->
        error("smoke test", "Emily.Backend smoke test failed: #{reason}",
          guidance: [
            "Fix any platform, artifact, or NIF failures above first.",
            "If those pass, verify Metal is available and retry with `mix emily.doctor`."
          ]
        )
    end
  end

  # `Code.ensure_loaded/1` triggers Emily.Native.__on_load__/0, which
  # calls `:erlang.load_nif/2`. A `{:module, _}` result therefore
  # proves dlopen succeeded and all symbols registered; we deliberately
  # do NOT call create_worker/0 here because it would leak an MLX
  # worker thread until the next BEAM GC, and the smoke check below
  # exercises the worker path via supervised Emily.MlxStream.
  defp load_nif do
    case Code.ensure_loaded(Emily.Native) do
      {:module, Emily.Native} ->
        {:ok, "Emily.Native loaded"}

      {:error, reason} ->
        {:error, inspect(reason)}
    end
  end

  defp smoke_test do
    case Application.ensure_all_started(:emily) do
      {:ok, _apps} ->
        t = Nx.tensor([1.0, 2.0], type: {:f, 32}, backend: Emily.Backend)
        result_tensor = t |> Nx.add(1.0) |> Nx.sum()
        result = Nx.to_number(result_tensor)

        cond do
          not match?(%Nx.Tensor{data: %Emily.Backend{}}, result_tensor) ->
            {:error,
             "result tensor left Emily.Backend (got #{inspect(result_tensor.data.__struct__)}) — backend fell back"}

          result != 5.0 ->
            {:error, "expected 5.0, got #{inspect(result)}"}

          true ->
            {:ok, "Emily.Backend produced 5.0 for a tiny tensor"}
        end

      {:error, reason} ->
        {:error, "could not start :emily: #{inspect(reason)}"}
    end
  end

  defp safe_call(fun) do
    case fun.() do
      :ok -> {:ok, "ok"}
      {:ok, summary} -> {:ok, summary}
      {:error, reason} -> {:error, inspect_reason(reason)}
      other -> {:error, "unexpected result #{inspect(other)}"}
    end
  rescue
    exception -> {:error, Exception.message(exception)}
  catch
    :exit, reason -> {:error, "exit #{inspect(reason)}"}
    kind, reason -> {:error, "#{kind} #{inspect(reason)}"}
  end

  defp priv_dir do
    case :code.priv_dir(:emily) do
      path when is_list(path) -> List.to_string(path)
      {:error, _reason} -> :error
    end
  end

  defp macos_version do
    case :os.type() do
      {:unix, :darwin} ->
        case System.cmd("sw_vers", ["-productVersion"], stderr_to_stdout: true) do
          {version, 0} -> String.trim(version)
          _ -> :unknown
        end

      _ ->
        :unknown
    end
  rescue
    _ -> :unknown
  end

  defp parse_macos_version(:unknown), do: :unknown
  defp parse_macos_version(nil), do: :unknown

  defp parse_macos_version(version) when is_binary(version) do
    parts =
      version
      |> String.split(".")
      |> Enum.take(3)
      |> Enum.map(&Integer.parse/1)

    case parts do
      [{major, ""}, {minor, ""} | rest] ->
        patch =
          case rest do
            [{patch, ""} | _] -> patch
            _ -> 0
          end

        {major, minor, patch}

      [{major, ""}] ->
        {major, 0, 0}

      _ ->
        :unknown
    end
  end

  defp parse_macos_version(_version), do: :unknown

  defp version_too_old?(:unknown, _variant), do: false

  defp version_too_old?(version, variant) do
    {minimum, _label} = minimum_macos(variant)
    version < minimum
  end

  defp minimum_macos(:jit), do: {@jit_min_macos, ":jit"}
  defp minimum_macos(_variant), do: {@aot_min_macos, ":aot"}

  defp apple_silicon?(arch) do
    String.starts_with?(arch, ["aarch64", "arm64"])
  end

  defp format_check(check) do
    lines = ["[#{check.status}] #{check.name}: #{check.summary}"]

    lines =
      Enum.reduce(check.guidance, lines, fn guidance, acc ->
        acc ++ ["    - #{guidance}"]
      end)

    Enum.join(lines, "\n")
  end

  defp format_version({major, minor, 0}), do: "#{major}.#{minor}"
  defp format_version({major, minor, patch}), do: "#{major}.#{minor}.#{patch}"

  defp ok(name, summary), do: %{name: name, status: :ok, summary: summary, guidance: []}
  defp skip(name, summary), do: %{name: name, status: :skip, summary: summary, guidance: []}

  defp warn(name, summary, opts) do
    %{name: name, status: :warn, summary: summary, guidance: Keyword.fetch!(opts, :guidance)}
  end

  defp error(name, summary, opts) do
    %{name: name, status: :error, summary: summary, guidance: Keyword.fetch!(opts, :guidance)}
  end

  defp inspect_reason(reason) when is_binary(reason), do: reason
  defp inspect_reason(reason), do: inspect(reason)

  defp count(report, status), do: Enum.count(report.checks, &(&1.status == status))
end
