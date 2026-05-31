defmodule Mix.Tasks.Emily.Checksums do
  @shortdoc "Pin SHA-256 checksums for the precompiled NIF tarballs"

  @moduledoc """
  Regenerate `native_checksums.txt` for the current `@version`.

  Hex consumers verify each downloaded NIF tarball against the checksum
  pinned in `native_checksums.txt`, which ships inside the hex package
  (and is therefore covered by Hex's package hash in the consumer's
  `mix.lock`). That roots trust independently of the GitHub release the
  tarball is fetched from.

  Run as part of cutting a release, *after* `release-nif.yml` has built
  and uploaded the tarballs and the draft release is public. Normally you
  don't invoke this directly — `mix emily.publish` runs it before `mix
  hex.publish`. The generated file is git-ignored: it is packaged at
  publish time, not committed.

  It downloads each supported tarball from the release and computes the
  checksum locally (it does not trust the `.sha256` sidecars).
  """

  use Mix.Task

  # Mirrors @supported_targets in mix.exs.
  @targets [{"aot", "macos-arm64"}, {"jit", "macos-arm64"}]

  @output "native_checksums.txt"

  @impl Mix.Task
  def run(_args) do
    {:ok, _} = Application.ensure_all_started(:inets)
    {:ok, _} = Application.ensure_all_started(:ssl)

    config = Mix.Project.config()
    version = config[:version]
    source_url = config[:source_url]

    lines =
      for {variant, target} <- @targets do
        asset = "emily-nif-#{version}-#{variant}-#{target}.tar.gz"
        url = "#{source_url}/releases/download/#{version}/#{asset}"
        Mix.shell().info("Fetching #{asset}")
        "#{Emily.NifArtifact.sha256_hex(download!(url))}  #{asset}"
      end

    File.write!(@output, header(version) <> Enum.join(lines, "\n") <> "\n")
    Mix.shell().info("Wrote #{@output} for #{version}")
  end

  defp header(version) do
    """
    # SHA-256 checksums for emily's precompiled NIF tarballs, pinned in
    # the hex package so consumers verify downloads against a trust root
    # independent of the (mutable) GitHub release. Regenerate with
    # `mix emily.checksums` after release-nif.yml builds the artifacts
    # for emily #{version}.
    """
  end

  defp download!(url) do
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

    case :httpc.request(:get, {String.to_charlist(url), []}, http_opts, body_format: :binary) do
      {:ok, {{_, 200, _}, _headers, body}} ->
        body

      {:ok, {{_, status, reason}, _headers, _body}} ->
        Mix.raise("checksum fetch failed (HTTP #{status} #{reason}): #{url}")

      {:error, reason} ->
        Mix.raise("checksum fetch failed (#{inspect(reason)}): #{url}")
    end
  end
end
