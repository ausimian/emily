#!/usr/bin/env bash
# Smoke-test the Hex package: build the tarball, unpack it, consume it from
# a throwaway project via a path dep, and exercise Emily.Backend end-to-end.
# Shares the real ~/Library/Caches/emily MLX build cache so repeat runs are
# fast — pass EMILY_CACHE=/path/to/dir to isolate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="$(mktemp -d -t emily-smoke.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

cd "$REPO_ROOT"

echo "==> Building Hex package"
rm -f emily-*.tar
mix hex.build
TARBALL="$(ls emily-*.tar | head -1)"
cp "$TARBALL" "$WORK_DIR/"

echo "==> Unpacking $TARBALL into $WORK_DIR/package"
mkdir -p "$WORK_DIR/package"
(cd "$WORK_DIR" && tar xf "$TARBALL" && tar xzf contents.tar.gz -C package)

echo "==> Creating consumer project"
CONSUMER="$WORK_DIR/consumer"
mkdir -p "$CONSUMER/lib"

cat > "$CONSUMER/mix.exs" <<'MIX'
defmodule EmilySmoke.MixProject do
  use Mix.Project

  def project do
    [app: :emily_smoke, version: "0.0.1", elixir: "~> 1.18", deps: deps()]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps, do: [{:emily, path: "../package"}]
end
MIX

cat > "$CONSUMER/lib/smoke.ex" <<'SMOKE'
defmodule EmilySmoke do
  import Nx.Defn

  defn add_mul(a, b), do: (a + b) * 2

  def run do
    Nx.default_backend(Emily.Backend)
    Nx.Defn.default_options(compiler: Emily.Compiler)

    a = Nx.tensor([1.0, 2.0, 3.0])
    b = Nx.tensor([4.0, 5.0, 6.0])

    sum = a |> Nx.add(b) |> Nx.to_flat_list()
    expected_sum = [5.0, 7.0, 9.0]
    if sum != expected_sum do
      raise "backend add mismatch: got #{inspect(sum)}, want #{inspect(expected_sum)}"
    end

    compiled = add_mul(a, b) |> Nx.to_flat_list()
    expected_compiled = [10.0, 14.0, 18.0]
    if compiled != expected_compiled do
      raise "defn mismatch: got #{inspect(compiled)}, want #{inspect(expected_compiled)}"
    end

    IO.puts("Emily.Backend add      -> #{inspect(sum)}")
    IO.puts("Emily.Compiler defn    -> #{inspect(compiled)}")
  end
end
SMOKE

cd "$CONSUMER"

echo "==> Fetching deps"
mix deps.get

echo "==> Compiling (first run builds MLX from source)"
mix compile

echo "==> Exercising Emily"
mix run -e 'EmilySmoke.run()'

echo "==> Smoke test passed"
