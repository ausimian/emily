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
  `~/Library/Caches/emily/mlx-<v>-<variant>`. `elixir_make` then
  compiles `c_src/*.cpp` against that MLX install and links
  `priv/libemily.{so,dylib}`.

- **Hex consumer (no `c_src/` in the tarball).** `mix compile` runs
  `:emily_nif`. The `compile.emily_nif` alias downloads the matching
  `emily-nif-<v>-<variant>-<target>.tar.gz` from the emily GitHub
  release for the tag, verifies its SHA256 against `@nif_checksums`
  in mix.exs, and extracts into `priv/`. No compilation; no MLX
  source tree on the consumer side.

The switch is driven by a `File.dir?("c_src")` check in mix.exs's
`compilers/0` — the hex `package[:files]` list ships only `lib/` and
the docs, so consumers land on the download path automatically.

Variant selection is unified via the `:variant` app-config key:
in-repo builds read `EMILY_MLX_VARIANT` env var (`aot`|`jit`,
default `aot`) through `config/config.exs` and stash the atom as
`Application.get_env(:emily, :variant)`; hex consumers set
`config :emily, variant: :jit` in their own `config/config.exs`.

## Cutting a release

### 1. Land changes on `main`

Normal PR flow. The per-matrix CI lane (`precommit` job) is the
canonical "still works" signal.

### 2. Tag and push

```sh
mix publisho patch   # or minor / major
```

This bumps `@version`, rolls `RELEASE.md` into `CHANGELOG.md` under a
dated heading, commits, tags (bare semver — no `v` prefix), and
pushes both the commit and the tag.

### 3. Wait for `release-nif.yml`

The tag push fires `.github/workflows/release-nif.yml`, which fans
out `{variant × target}`:

| Variant | Target       | Runner      |
| ------- | ------------ | ----------- |
| aot     | macos-arm64  | `macos-14`  |
| jit     | macos-arm64  | `macos-26`  |

Each cell:

1. Clones `:mlx_src` via `mix deps.get`.
2. Builds MLX + the NIF from source (`scripts/build-mlx.sh` +
   `elixir_make`).
3. Tars `priv/libemily.* + priv/mlx.metallib` as
   `emily-nif-<v>-<variant>-<target>.tar.gz`.
4. Uploads the tarball + `.sha256` to a draft release named after
   the tag.
5. Prints a ready-to-paste `@nif_checksums` line in the job summary.

### 4. Bake the SHAs into `mix.exs`

Copy the `{:aot, "macos-arm64"} => "..."` lines from each job
summary into `@nif_checksums` in `mix.exs`. Commit as
`Version <v> checksums`.

### 5. Re-tag and re-run

`mix publisho patch` again to cut a second tag (e.g. if the draft
was `0.3.0`, this gives you `0.3.1` with the checksums baked in).
`release-nif.yml` fires a second time, rebuilds the same tarballs,
and uploads them to a fresh draft.

If the inputs are unchanged (`mix.lock`, `c_src/**`, `Makefile`,
`scripts/build-mlx.sh`, pinned `@mlx_version`), the rebuilt tarballs
should hash identically to the baked-in SHAs — paste-check before
publishing.

> The double-tag dance is awkward but aligns with the existing
> "tag-then-checksum-then-tag" pattern from `@mlx_checksums`. A
> future `mix emily.checksum` helper (rustler_precompiled-style)
> could download uploaded artefacts and update `@nif_checksums`
> in-place, collapsing this to one manual step.

### 6. Verify the published tarball end-to-end

In a throwaway project:

```sh
mix new /tmp/emily-verify && cd /tmp/emily-verify
# add {:emily, "~> <v>"} to deps
mix deps.get && mix compile
iex -S mix
# Nx.default_backend(Emily.Backend)
# Nx.tensor([1.0, 2.0]) |> Nx.add(3) |> Nx.to_flat_list()
```

### 7. Promote the draft and publish

```sh
gh release edit <v> --repo ausimian/emily --draft=false
mix hex.publish
```

## Bumping MLX

Emily pins an MLX version in `mix.exs` (`@mlx_version`). The
`:mlx_src` git dep is cloned at `v<@mlx_version>` by `mix deps.get`,
so changing the attribute is the entire pin.

1. Bump `@mlx_version` in mix.exs.
2. `mix deps.update mlx_src`.
3. Force a local MLX rebuild to sanity-check:
   ```sh
   rm -rf ~/Library/Caches/emily/mlx-<new>-*
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

The consumer will hit the real `compile.emily_nif` step — if
`@nif_checksums` is populated and the tarball is on the GitHub
release, it downloads + extracts; otherwise you get a clear
"No precompiled NIF pinned" error.

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
