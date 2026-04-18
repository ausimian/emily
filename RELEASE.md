# Release notes for next release

## Added

- M23 — Public documentation & examples review. Pre-1.0 pass over the
  documentation surface users actually consume (moduledocs, README,
  HexDocs navigation, worked examples).
  - **`notebooks/distilbert_qa.livemd`** — DistilBERT QA pipeline
    wired through `Emily.Compiler`; mirrors the conformance suite
    without the test harness. `Mix.install/2` at the top so the
    notebook runs standalone.
  - **`notebooks/qwen3_quantized.livemd`** — Qwen3-0.6B loaded,
    int4-quantized group-wise, greedy-decoded under
    `Bumblebee.Text.generation/4`. Demonstrates `Emily.Stream` for
    concurrent per-process Metal command queues.
  - **`mix.exs` docs config**: `groups_for_modules` organises the
    HexDocs nav by concern (Core / Concurrency / Quantization /
    Training / Performance / Observability); `extras` adds the
    two notebooks under a `Notebooks` group.
  - **Moduledoc pass**. Removed stale milestone references; clarified
    the "Public API" vs Nx-callback-impl split on `Emily.Backend`
    and `Emily.Compiler`; reorganised `Divergences from
    Nx.BinaryBackend` into a dedicated subsection. Added `iex>`
    examples to `Emily`, `Emily.Compiler`, `Emily.Stream`,
    `Emily.Quantization`, `Emily.QuantizedWeight`,
    `Emily.MixedPrecision`, and `Emily.Telemetry`.
  - **README**: new `Documentation` section with a HexDocs link and
    notebook pointers. The stale "Milestones shipped" section was
    replaced by a brief summary that points at `CHANGELOG.md` for
    the per-milestone breakdown. Testing commands updated to
    include `:qwen3_quant_full`, `:training_full`,
    `:distilbert_full`.
  - **CHANGELOG cutover**: moved the full M0–M22 release history
    from `RELEASE.md` into `CHANGELOG.md` under
    `## [0.1.0] - unreleased`. `RELEASE.md` now carries only the
    M23 notes, matching the conventions' "release notes for next
    release" contract.
  - **`mix docs` runs clean** over `lib/*.ex` — no unresolved
    cross-refs from moduledocs. Historical references in
    `CHANGELOG.md` / `PLAN.md` to hidden internal modules
    (`Emily.Native` etc.) remain as prose; they are accurate
    descriptions of the state at the time the entry was written.
