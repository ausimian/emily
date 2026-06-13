#!/usr/bin/env bash
# Smoke-test the example livebooks against the LOCAL emily checkout.
#
# For each notebook in livebooks/, this extracts its Elixir cells,
# repoints the `{:emily, "~> x"}` Mix.install dependency at this repo (as
# a `path:` dep, so the notebook exercises the working tree — including a
# from-source NIF build), and runs the result headlessly with `elixir`.
# A notebook passes if it runs to completion with exit code 0.
#
# Notes:
#   * First run downloads model checkpoints via Bumblebee (cached under
#     ~/Library/Caches/bumblebee afterwards) and compiles emily's NIF, so
#     it needs network + disk and can take several minutes per notebook.
#   * Cells render Kino widgets; outside Livebook those just evaluate to
#     structs. Notebooks are *run*, not interacted with — anything gated
#     behind a Kino form/input (e.g. recording audio in the Whisper
#     notebook) won't fire, but the surrounding setup/inference still does.
#
# Usage:
#   scripts/test-livebooks.sh                                  # all notebooks
#   scripts/test-livebooks.sh distilbert_qa nomic_embeddings   # a subset
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

# Print the contents of every ```elixir fenced block, blank-line separated.
extract_cells() {
  awk '
    /^```elixir[[:space:]]*$/ { inblock=1; next }
    /^```[[:space:]]*$/       { if (inblock) { inblock=0; print "" }; next }
    inblock                   { print }
  ' "$1"
}

pass=()
fail=()
skip=()
log_dir="$(mktemp -d "${TMPDIR:-/tmp}/emily-livebooks.XXXXXX")"
echo "emily: $REPO (path dep)"
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
  # Extract the cells and repoint the emily dependency at this checkout.
  extract_cells "$nb" \
    | sed "s|{:emily,[^}]*}|{:emily, path: \"$REPO\"}|" \
    > "$script"

  printf ">> %-28s running (timeout %ss) ... " "$name" "$TIMEOUT"
  start=$SECONDS
  if timeout "$TIMEOUT" elixir "$script" >"$out" 2>&1; then
    echo "PASS ($((SECONDS - start))s)"
    pass+=("$name")
  else
    code=$?
    echo "FAIL (exit $code, $((SECONDS - start))s)"
    tail -n 25 "$out" | sed 's/^/   | /'
    fail+=("$name")
  fi
done

echo
echo "==================== livebook results ===================="
printf "PASS (%d): %s\n" "${#pass[@]}" "${pass[*]:-}"
printf "FAIL (%d): %s\n" "${#fail[@]}" "${fail[*]:-}"
printf "SKIP (%d): %s\n" "${#skip[@]}" "${skip[*]:-}"
echo "logs in $log_dir"

[[ ${#fail[@]} -eq 0 ]]
