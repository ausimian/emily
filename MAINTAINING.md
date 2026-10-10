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
`native_checksums.txt`, which ships in the hex package. Once the tag is
pushed, the release workflow builds the NIFs, publishes the GitHub
release, regenerates that file from the published tarballs and publishes
the package to Hex, so there is nothing to update or commit by hand (the
file is git-ignored and can't go stale).

### 1. Land changes on `main`

Normal PR flow. The per-matrix CI lane (`precommit` job) is the
canonical "still works" signal.

### 2. Bump `@version`, roll the changelog, tag

```sh
mix publisho patch   # or minor / major
```

Run it on a `release/<v>` branch cut from the tip of `main`: `main` is
protected and takes changes through PRs only. `publisho` bumps
`@version`, rolls `RELEASE.md` into `CHANGELOG.md` under a dated
`## <v>` heading, commits `Version <v>`, and creates an annotated
bare-semver tag (no `v` prefix). It does **not** push — that's the next
step.

### 3. Merge the release PR, then push the tag

```sh
git push -u origin release/<v>   # the branch only, not the tag
gh pr create --base main --title "Version <v>"
# once CI is green, merge it with a merge commit, then:
git push origin <v>              # the tag — this is what fires the release workflow
```

Push the tag only after the PR merges, so the release builds from a
commit in `main`'s history. A merge commit (not a squash) keeps the
tagged `Version <v>` commit itself in that history. The workflow's
`prepare-release` job refuses a tag whose commit isn't in `main`'s
history, so a tag pushed too early fails before anything is built or
released.

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

Once both cells finish, the workflow's `publish-release` job flips the
release out of draft **automatically** (`gh release edit <v> --draft=false`),
so its assets become public with no manual step. That job is gated to the
tag-push path, so a manual `workflow_dispatch` rebuild leaves the release
untouched — see *Rebuilding without retagging*.

### 4. Hex publish (automatic)

When the release is public, the workflow's `publish-hex` job:

1. checks that the tag matches mix.exs's `@version`, since Hex publishes
   `@version`;
2. runs `mix emily.publish` (an alias for `mix emily.checksums`), which
   downloads each tarball from the public release and records its SHA256
   in `native_checksums.txt`. It does **not** trust the `.sha256`
   sidecars; it hashes the bytes itself;
3. runs `mix hex.build`, then `mix hex.publish --yes`, publishing the
   package (including `native_checksums.txt`) and its docs.

Because `native_checksums.txt` ships inside the package, it is covered
by Hex's package hash in the consumer's `mix.lock`, a trust root
independent of the mutable GitHub release.

It authenticates with `HEX_API_KEY`, a secret on the repository's `hex`
environment. One-time setup:

1. Generate a Hex API key with write access (on hex.pm, under
   Dashboard → Keys), scoped as narrowly as Hex allows.
2. In the repository settings, create an environment named `hex`. Under
   *Deployment branches and tags*, allow only tags matching
   `[0-9]*.[0-9]*.[0-9]*`.
3. Add the key to that environment as the secret `HEX_API_KEY`.
4. Under *Settings → Rules → Rulesets*, create an active tag ruleset
   ("Release tags") targeting `[0-9]*.[0-9]*.[0-9]*`. Restrict creations,
   updates and deletions, with *Repository admin* as the only bypass.

The environment rule matches tag *names* only, and a tag-push run uses
the workflow file from the tagged commit. So the ruleset is what keeps
the key safe: without it, anyone able to push a matching tag could
publish an arbitrary commit, and no check inside the workflow could
stop them. With it, only repository admins can create release tags, and
only the tag-triggered run uses the `hex` environment, so no other run
can read the key.

A `workflow_dispatch` run executes the same job without the upload: it
pins checksums for the current `@version`'s (already public) release,
then runs `mix hex.build` and `mix docs`. Use it to check the job before
tagging. (`mix hex.publish --dry-run` can't stand in for it, because it
still requires a key.)

#### Publishing by hand

If `publish-hex` fails after the release is public, fix the cause and
re-run the job from the Actions tab. Pinning is idempotent, and
`hex.publish` refuses to overwrite a published version without
`--replace`. To publish from a checkout instead:

```sh
mix hex.user auth    # if this machine isn't authenticated
mix emily.publish    # pin checksums from the public release
mix hex.publish      # package (incl. native_checksums.txt) + docs
```

These are two separate commands on purpose — they can't be folded into
one alias. Mix only loads the Hex archive for the task *named on the
command line*, so a `hex.publish` step chained inside an alias (whose CLI
name is `emily.publish`) fails with `** (Mix) The task "hex.publish" could
not be found`. Running `mix hex.publish` directly is what loads Hex.
`hex.publish` keeps its `:docs` `preferred_env`, so it publishes docs as
well as the package.

### 5. Verify end-to-end (recommended)

After publishing, in a throwaway project:

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
the first release of a bump. The checksum pin only exists in the
*published* package, so this must run against the published package,
i.e. after step 4.

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
on that version see no change. The `publish-hex` job runs too, but
stops short of uploading (see step 4), so nothing reaches Hex.

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

### Static analysis of the NIF (cppcheck)

```sh
brew install cppcheck   # one-time
make cppcheck
```

Runs cppcheck over `c_src/` and exits non-zero on any finding. It needs
neither a built `libmlx` nor the BEAM toolchain, so it works on a bare
checkout in seconds — the same target CI runs (`.github/workflows/ci.yml`,
the `cppcheck` job). See the `cppcheck` target in the `Makefile` for the
enabled checks and suppressions; use inline `// cppcheck-suppress <id>`
for one-off false positives.

### Static analysis of the NIF (clang-tidy)

```sh
brew install llvm   # one-time; clang-tidy isn't in stock Xcode
mix clang.tidy
```

Runs clang-tidy (including the clang static analyzer, via its
`clang-analyzer-*` checks) over `c_src/` and exits non-zero on any finding
— the same tool the `clang-tidy` CI job runs. Unlike cppcheck it compiles
each translation unit, so it needs the MLX/Fine/ERTS headers and build
flags; `mix clang.tidy` supplies that env (reusing the cached MLX) and
drives the `clang-tidy` Makefile target, so `make clang-tidy` on its own
will refuse to run. Enabled checks and the header filter live in the
repo-root `.clang-tidy`; use inline `// NOLINT(<check>)` for one-off false
positives. Point at a specific binary with `CLANG_TIDY=/path/to/clang-tidy`.

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
