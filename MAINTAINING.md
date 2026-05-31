# Maintaining Emily

Maintainer-facing runbook for tasks that don't fit in the consumer-facing
README. If you're just *using* Emily, start at `README.md`.

## How the build is wired

Emily has two distinct compile paths depending on whether it's being
built from source (in this repo / CI) or consumed as a hex package:

- **In-repo / CI (has `c_src/`).** `mix compile` runs
  `:emily_mlx → :elixir_make`. The `compile.emily_mlx` alias calls
  `scripts/build-mlx.sh`, which cmake-builds libmlx.a + mlx.metallib
  from the `:mlx_src` Mix git dep (`deps/mlx_src/`) and installs into
  `$EMILY_CACHE/mlx-<v>-<variant>` (default
  `$(getconf DARWIN_USER_CACHE_DIR)emily/mlx-<v>-<variant>` on macOS,
  `${XDG_CACHE_HOME:-~/.cache}/emily/mlx-<v>-<variant>` on Linux).
  `elixir_make` then compiles `c_src/*.cpp` against that MLX install
  and links `priv/libemily.{so,dylib}`.

- **Hex consumer (no `c_src/` in the tarball).** `mix compile` runs
  `:emily_nif`. The `compile.emily_nif` alias downloads the matching
  `emily-nif-<v>-<variant>-<target>.tar.gz` from the emily GitHub
  release for the tag, verifies its SHA256 against the checksum pinned
  in `native_checksums.txt` (shipped *inside* the hex package, so it's
  covered by Hex's package hash in the consumer's `mix.lock` — a trust
  root independent of the mutable release), validates the archive
  entries against an allowlist (`libemily.so`/`libemily.dylib` +
  `mlx.metallib`; rejects symlinks, hardlinks, `..` traversal, absolute
  paths, and any unexpected entry), and extracts those files into
  `priv/` via `:erl_tar`. No compilation; no MLX source tree on the
  consumer side. See `Emily.NifArtifact`.

The switch is driven by a `File.dir?("c_src")` check in mix.exs's
`compilers/0` — the hex `package[:files]` list ships only `lib/` and
the docs, so consumers land on the download path automatically.

`EMILY_CACHE` must point at a **private, user-owned** directory: the
source build statically links `$EMILY_CACHE/mlx-<v>-<variant>/lib/libmlx.a`
into the NIF, so the build refuses to reuse a cache dir owned by another
user (and keeps its own dirs `0700`) to stop a shared cache from planting
native code. The per-user macOS/XDG defaults already satisfy this.

Variant selection is unified via the `:variant` app-config key:
in-repo builds read `EMILY_MLX_VARIANT` env var (`aot`|`jit`,
default `aot`) through `config/config.exs` and stash the atom as
`Application.get_env(:emily, :variant)`; hex consumers set
`config :emily, variant: :jit` in their own `config/config.exs`.

## Cutting a release

Consumers verify each NIF tarball against the checksum pinned in
`native_checksums.txt`, which ships in the hex package. The
`mix emily.publish` alias regenerates that file from the freshly-built
release artifacts before publishing (step 4), so there is nothing to
update or commit by hand — the file is git-ignored and can't go stale.

### 1. Land changes on `main`

Normal PR flow. The per-matrix CI lane (`precommit` job) is the
canonical "still works" signal.

### 2. Bump `@version`, roll the changelog, tag

```sh
mix publisho patch   # or minor / major
```

Bumps `@version`, rolls `RELEASE.md` into `CHANGELOG.md` under a
dated `## <v>` heading, commits `Version <v>`, tags bare semver (no
`v` prefix), and pushes both the commit and the tag.

The tag push fires `.github/workflows/release-nif.yml`, which fans
out `{variant × target}`:

| Variant | Target       | Runner      |
| ------- | ------------ | ----------- |
| aot     | macos-arm64  | `macos-14`  |
| jit     | macos-arm64  | `macos-26`  |

Each cell clones `:mlx_src`, builds MLX + the NIF from source
(`scripts/build-mlx.sh` + `elixir_make`), tars
`priv/libemily.* + priv/mlx.metallib` as
`emily-nif-<v>-<variant>-<target>.tar.gz`, writes a `.sha256`
sidecar (informational — consumers verify against the pinned
`native_checksums.txt`, not the sidecar), and uploads both to a
**draft** GitHub release at
`https://github.com/ausimian/emily/releases/tag/<v>` — the URL the
consumer's `compile.emily_nif` step fetches from.

### 3. Verify end-to-end

In a throwaway project:

```sh
mix new /tmp/emily-verify && cd /tmp/emily-verify
# add {:emily, "~> <v>"} to deps
mix deps.get && mix compile
iex -S mix
# Nx.default_backend(Emily.Backend)
# Nx.tensor([1.0, 2.0]) |> Nx.add(3) |> Nx.to_flat_list()
```

`mix compile` reads the pinned checksum from `native_checksums.txt`,
downloads the tarball, verifies, validates entries, extracts. A
variant-mismatched consumer (`config :emily, variant: :jit`) should
download the JIT tarball instead — worth spot-checking both lanes on
the first release of a bump. (The pin only ever exists in the
*published* package — `native_checksums.txt` is git-ignored and
generated during `mix hex.publish` — so run this end-to-end verify
against the published package, i.e. after step 4.)

### 4. Promote the draft and publish

Promote the release so its assets are public, then publish:

```sh
gh release edit <v> --repo ausimian/emily --draft=false   # assets go public
mix emily.publish                                          # regenerate checksums, then publish
```

`mix emily.publish` runs `mix emily.checksums` first: it downloads each
tarball from the (now-public) release, records its SHA256 into
`native_checksums.txt`, and `mix hex.publish` then packages that file.
(Use `mix emily.publish`, not a plain `mix hex.publish` — the latter
skips the regen, though it fails closed since `hex.build` errors on the
missing checksums file.)
So the consumer verifies downloads against a trust root that lives in
the immutable Hex package, not the mutable GitHub release — with no file
to maintain and nothing to commit. The file is git-ignored and
regenerated on every publish, so it can't go stale. If the draft isn't
public yet, `mix emily.checksums` 404s and aborts the publish.

### Rebuilding without retagging

If you need to reproduce a release's artefacts out-of-band (say, to
compare against an earlier build, or to iterate on
`scripts/build-mlx.sh` without bumping the version), trigger
`release-nif.yml` manually:

```sh
gh workflow run release-nif.yml --repo ausimian/emily --ref main
```

The dispatch run resolves `@version` from `mix.exs`, builds the
same tarballs, and stashes them as workflow-run artefacts
(retention 90 days). The GitHub release is untouched, so consumers
on that version see no change.

## Bumping MLX

Emily pins an MLX version in `mix.exs` (`@mlx_version`). The
`:mlx_src` git dep is cloned at `v<@mlx_version>` by `mix deps.get`,
so changing the attribute is the entire pin.

1. Bump `@mlx_version` in mix.exs.
2. `mix deps.update mlx_src`.
3. Force a local MLX rebuild to sanity-check:
   ```sh
   rm -rf "$(getconf DARWIN_USER_CACHE_DIR)emily/"mlx-<new>-*   # macOS default
   # or: rm -rf "${EMILY_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/emily}/"mlx-<new>-*
   mix precommit
   ```
4. Note the bump in `RELEASE.md`.
5. Land the PR, then follow the release flow above. CI's NIF builds
   pick up the new MLX automatically.

## Local debugging

### Build MLX in isolation

```sh
mix deps.get        # populate deps/mlx_src
scripts/build-mlx.sh deps/mlx_src <v> 0 /tmp/mlx-install   # 0 = AOT, 1 = JIT
```

### Simulate the hex-consumer path locally

```sh
mix hex.build                    # produces emily-<v>.tar
# unpack into a throwaway project as a path dep
# (see prior scripts/smoke-test-package.sh for the pattern)
```

The consumer will hit the real `compile.emily_nif` step — if the
tarball and its `.sha256` sidecar are present on the published
GitHub release for the tag, it downloads + verifies + extracts;
otherwise the sidecar fetch 404s with a clear `NIF download failed
(HTTP 404 Not Found)` error pointing at the missing asset URL.

## Why the JIT lane can't roam across macOS versions

The JIT `libmlx.a` is built against the macOS 26.2+ SDK — MLX's NAX
kernel sources transitively include
`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, which
only ships in that SDK, and they also end up referencing libSystem
symbols (e.g. `__fmaxf16`) that older macOS releases don't have. The
JIT NIF therefore requires macOS 26.2+ at runtime as well as build
time, which is why the JIT CI lane runs on `macos-26` — the binary
won't dlopen on older hosts.

The AOT lane has no such constraint and is built on `macos-14`, so
the AOT NIF runs anywhere from macOS 14 upward.
