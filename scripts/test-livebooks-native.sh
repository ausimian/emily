#!/usr/bin/env bash
# Test whether the example livebooks lower FULLY NATIVE under the
# single-NIF Expr compiler.
#
# A clone of `test-livebooks.sh` that, in addition to repointing the
# `{:emily, "~> x"}` Mix.install dependency at this checkout, forces every
# `Nx.Defn` computation through `compiler: Emily.Compiler, native: true`
# with `native_fallback: :raise`. That makes each livebook a no-fallback
# probe: it runs to completion (exit 0) ONLY if every op it compiles lowers
# to the native program. The first op that can't lower RAISES — the failing
# tail names it (`... does not yet lower op :foo`) — so this reports, per
# livebook, whether it can be converted to native and, if not, what's
# missing.
#
# How the forcing works: a preamble injected right after the first cell
# (`Mix.install`) calls `Nx.Defn.global_default_options/1` (process-GLOBAL
# via persistent_term, so `Nx.Serving` workers / `Task`s inherit it) and
# sets `config :emily, native_fallback: :raise`. Per-call options that pass
# only `compiler: Emily.Compiler` merge with the global `native: true`, so
# they're upgraded to native too.
#
# Notes:
#   * Same caveats as `test-livebooks.sh`: first run downloads checkpoints
#     and builds emily's NIF from source; Kino widgets just evaluate to
#     structs (forms/inputs don't fire), so anything gated behind a Kino
#     form (e.g. the Whisper live-recording cell) won't run — but the
#     surrounding setup/inference still does.
#   * A livebook that PASSES here is one whose every compiled op lowers
#     native. A FAIL is either a genuine error or — the interesting case —
#     an un-lowerable op raising under `:raise`.
#
# Usage:
#   scripts/test-livebooks-native.sh                                  # all
#   scripts/test-livebooks-native.sh distilbert_qa nomic_embeddings   # subset
#
# Env:
#   LIVEBOOK_TIMEOUT   per-notebook timeout, seconds (default 1200)
#   LIVEBOOK_SKIP      space-separated notebook names to skip
#   EMILY_CACHE        MLX/NIF cache dir (passed through to the build)

set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NB_DIR="$REPO/livebooks"
TIMEOUT="${LIVEBOOK_TIMEOUT:-1200}"
SKIP=" ${LIVEBOOK_SKIP:-} "

if ! command -v elixir >/dev/null 2>&1; then
  echo "error: elixir not on PATH" >&2
  exit 2
fi

# Selection: explicit notebook basenames, or every *.livemd.
if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=()
  for f in "$NB_DIR"/*.livemd; do names+=("$(basename "$f" .livemd)"); done
fi

# Print the contents of every ```elixir fenced block, blank-line separated,
# and inject the native-forcing preamble right after the FIRST block
# (`Mix.install`, after which Nx/Emily are loaded so the global setters
# resolve).
extract_cells_native() {
  awk '
    /^```elixir[[:space:]]*$/ { inblock=1; next }
    /^```[[:space:]]*$/ {
      if (inblock) {
        inblock=0; print "";
        if (!injected) {
          print "# >>> forced native compilation (test-livebooks-native.sh)";
          print "Application.put_env(:emily, :native_fallback, :raise)";
          print "Nx.Defn.global_default_options(compiler: Emily.Compiler, native: true, native_fallback: :raise)";
          print "";
          injected=1;
        }
      }
      next
    }
    inblock { print }
  ' "$1"
}

pass=()
fail=()
skip=()
log_dir="$(mktemp -d "${TMPDIR:-/tmp}/emily-livebooks-native.XXXXXX")"
echo "emily: $REPO (path dep)"
echo "mode:  native: true, native_fallback: :raise (global default)"
echo "logs:  $log_dir"
echo

for name in "${names[@]}"; do
  nb="$NB_DIR/$name.livemd"
  if [[ ! -f "$nb" ]]; then
    echo ">> $name: NOT FOUND"
    fail+=("$name")
    continue
  fi
  if [[ "$SKIP" == *" $name "* ]]; then
    echo ">> $name: SKIP"
    skip+=("$name")
    continue
  fi

  script="$log_dir/$name.exs"
  out="$log_dir/$name.out"
  # Extract the cells, inject the native preamble after Mix.install, and
  # repoint the emily dependency at this checkout.
  extract_cells_native "$nb" \
    | sed "s|{:emily,[^}]*}|{:emily, path: \"$REPO\"}|" \
    > "$script"

  printf ">> %-28s native (timeout %ss) ... " "$name" "$TIMEOUT"
  start=$SECONDS
  if timeout "$TIMEOUT" elixir "$script" >"$out" 2>&1; then
    echo "NATIVE ($((SECONDS - start))s)"
    pass+=("$name")
  else
    code=$?
    echo "FAIL (exit $code, $((SECONDS - start))s)"
    # Surface the un-lowerable op / error. Prefer the compiler's own
    # "does not yet lower" line if present, else the failing tail.
    if grep -qiE "does not yet lower|cannot lower|not .*-compatible|fell back" "$out"; then
      grep -iE "does not yet lower|cannot lower|not .*-compatible|fell back" "$out" \
        | head -n 3 | sed 's/^/   ! /'
    fi
    tail -n 20 "$out" | sed 's/^/   | /'
    fail+=("$name")
  fi
done

echo
echo "================ livebook NATIVE results ================"
printf "NATIVE (%d): %s\n" "${#pass[@]}" "${pass[*]:-}"
printf "FAIL   (%d): %s\n" "${#fail[@]}" "${fail[*]:-}"
printf "SKIP   (%d): %s\n" "${#skip[@]}" "${skip[*]:-}"
echo "logs in $log_dir"

[[ ${#fail[@]} -eq 0 ]]
