defmodule Emily.NifArtifact do
  @moduledoc false
  # Pure helpers for verifying the precompiled NIF tarball that hex
  # consumers download at compile time.
  #
  # The trust root is `native_checksums.txt`, shipped *inside* the hex
  # package. Its contents are covered by Hex's own package hash recorded
  # in the consumer's `mix.lock`, so the expected checksum is rooted in
  # something the consumer already verifies — independent of the
  # (mutable) GitHub release the tarball itself is fetched from. An
  # attacker who can rewrite the release cannot change the pinned
  # checksum without failing the Hex integrity check.
  #
  # Used by `mix.exs`'s `fetch_nif/1` (the `:emily_nif` compiler) and by
  # `mix emily.checksums`. Deliberately side-effect-free so it can be
  # unit-tested in the default suite — the fetch path itself only runs on
  # the hex-consumer side and is never exercised by CI (which builds from
  # source).

  # The only entries a NIF tarball may contain. Anything else — a
  # symlink, hardlink, directory, an absolute path, a `..` traversal, or
  # an unexpected file — is rejected before extraction. (The macOS NIF is
  # `libemily.so`; `.dylib` is listed for forward-compatibility.)
  @allowlist ~w(libemily.so libemily.dylib mlx.metallib)

  @spec allowlist() :: [String.t()]
  def allowlist, do: @allowlist

  @doc "SHA-256 of a binary as lowercase hex."
  @spec sha256_hex(binary()) :: String.t()
  def sha256_hex(data) when is_binary(data) do
    :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)
  end

  @doc """
  Parse `sha256sum`-format text (`<hex>  <filename>` per line; blank
  lines and `#` comments ignored) into a `%{filename => hex}` map.
  """
  @spec parse_checksums(binary()) :: %{optional(String.t()) => String.t()}
  def parse_checksums(contents) when is_binary(contents) do
    contents
    |> String.split("\n")
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == "" or String.starts_with?(&1, "#")))
    |> Enum.flat_map(fn line ->
      case String.split(line, ~r/\s+/, parts: 2) do
        [hex, name] -> [{String.trim(name), String.downcase(hex)}]
        _ -> []
      end
    end)
    |> Map.new()
  end

  @doc "Look up the pinned checksum for `asset`."
  @spec expected(%{optional(String.t()) => String.t()}, String.t()) ::
          {:ok, String.t()} | :error
  def expected(checksums, asset), do: Map.fetch(checksums, asset)

  @doc """
  Validate a tarball's entries against the allowlist. `entries` is a list
  of `{name, type}` as produced from `:erl_tar.table/2` with `:verbose`.

  Rejects any non-regular entry (symlink, hardlink, directory, device)
  and any name not in `allowlist` — which also rejects absolute paths and
  `..` traversal, since those can never match an allowlisted basename.
  """
  @spec verify_entries([{String.t(), atom()}], [String.t()]) ::
          :ok | {:error, String.t()}
  def verify_entries(entries, allowlist \\ @allowlist) do
    Enum.reduce_while(entries, :ok, fn {name, type}, :ok ->
      cond do
        type != :regular ->
          {:halt, {:error, "refusing non-regular tar entry #{inspect(name)} (#{type})"}}

        name not in allowlist ->
          {:halt,
           {:error,
            "unexpected tar entry #{inspect(name)} (allowed: #{Enum.join(allowlist, ", ")})"}}

        true ->
          {:cont, :ok}
      end
    end)
  end
end
