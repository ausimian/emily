# Maintaining Emily

Maintainer-facing runbook for tasks that don't fit in the consumer-facing
README. If you're just *using* Emily, start at `README.md`.

## Bumping the MLX version

Emily pins an MLX version in `mix.exs` (`@mlx_version`) and consumes
prebuilt `libmlx.a` + `mlx.metallib` tarballs from this repo's releases
(`mlx-<version>` tags). To bump to a new MLX release:

### 1. Cut the prebuilts

Run the **Release MLX prebuilt** workflow (Actions → Run workflow) twice
— once per variant:

| Variant | Input `build_type` | Runner (auto-selected) |
| ------- | ------------------ | ---------------------- |
| AOT     | `no-jit`           | `macos-14`             |
| JIT     | `jit`              | `macos-26`             |

Each run uploads `mlx-<v>-macos-arm64-<variant>.tar.gz` + a `.sha256`
sidecar to a draft release tagged `mlx-<v>`. **Don't un-draft yet** —
until `mix.exs` references the new version and checksums, anyone
fetching the public release would get a mismatch.

### 2. Grab the checksums

```sh
gh release download mlx-<v> --repo ausimian/emily \
  --pattern '*.sha256' --dir /tmp/mlx-sha --clobber
cat /tmp/mlx-sha/*.sha256
```

### 3. Bump `mix.exs`

```elixir
@mlx_version "<v>"
@mlx_checksums %{
  "mlx-<v>-macos-arm64-aot.tar.gz" => "<new aot sha256>",
  "mlx-<v>-macos-arm64-jit.tar.gz" => "<new jit sha256>"
}
```

Both checksum map keys include the version — replace the whole map.

### 4. Optional housekeeping

- `.github/workflows/release-mlx.yml` — bump the `default:` on the
  `mlx_version` input so future manual dispatches default to the
  new version.
- `RELEASE.md` — add a "Changed" bullet noting the bump and anything
  notable from the MLX changelog.

### 5. Verify locally

```sh
rm -rf ~/Library/Caches/emily/mlx-<v>-*   # force cold download
mix precommit
```

`mix compile` fetches the new tarball, verifies the SHA256, extracts,
and the suite runs against it. A mistyped checksum aborts with a clear
mismatch error before the build.

### 6. PR → merge → un-draft

1. Open a PR; both CI lanes (`macos-14` / AOT, `macos-26` / JIT) must
   pass.
2. Merge.
3. Un-draft the MLX release so consumers can fetch:

   ```sh
   gh release edit mlx-<v> --repo ausimian/emily --draft=false
   ```

### Local debugging

`scripts/build-mlx-prebuilt.sh` produces a tarball locally with no
workflow round-trip — useful if you're tweaking cmake flags or the
packaging layout:

```sh
git clone --depth=1 --branch v<v> https://github.com/ml-explore/mlx /tmp/mlx-src
scripts/build-mlx-prebuilt.sh /tmp/mlx-src <v> 0   # 0 = AOT, 1 = JIT
```

Produces `mlx-<v>-macos-arm64-<variant>.tar.gz` + `.sha256` in the cwd.

### Why the JIT lane can't roam across macOS versions

The JIT `libmlx.a` is built against the macOS 26.2+ SDK — MLX's NAX
kernel sources transitively include
`<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`, which
only ships in that SDK, and they also end up referencing libSystem
symbols (e.g. `__fmaxf16`) that older macOS releases don't have. The
JIT prebuilt therefore requires macOS 26.2+ at runtime as well as
build time, which is why the JIT CI lane runs on `macos-26` — the
binary won't dlopen on older hosts.

The AOT lane has no such constraint and is built on `macos-14`, so
the AOT prebuilt runs anywhere from macOS 14 upward.
