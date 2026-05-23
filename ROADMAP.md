# Emily — roadmap

Active and future work. Shipped milestones live in [`PLAN.md`](https://github.com/ausimian/emily/blob/main/PLAN.md)
as the historical record; the current shape of the library is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## Goals

  * **Correctness over performance at every layer.** Every layer has
    its own oracle.
  * **Structural impossibility of bidirectional NIF deadlocks.** No
    C++→BEAM callbacks. No GenServer on the hot path.
  * **Bumblebee-first.** DistilBERT, Qwen3, ViT, Whisper, plus the
    Bumblebee 0.7 family (NomicBERT, SmolLM3, ModernBERT).
  * **Shippable at every milestone.** Backend-only mode is useful on
    its own; the Defn compiler is additive.

## Non-goals

  * Ahead-of-time compilation (`mlx::export` / IREE-style).
    Complementary, separate effort.
  * Windows or non-Apple-Silicon Linux GPU. CPU-only Linux is a
    nice-to-have for CI.
  * Distributed training (`mlx::distributed::*`), a native optimizer
    library, FSDP / ring allreduce. Autodiff + small-scale training
    loops are in scope; large-scale distributed is not.
  * Drop-in replacement for EMLX. We borrow where it's clearly right,
    but we're not constrained by its API.
  * User-level GPU kernel JIT from Elixir (`fast::metal_kernel` /
    `fast::cuda_kernel`). Orthogonal to Emily's "Nx backend, not a
    framework" stance.

## Deferred to post-1.0

Each line summarises a deferred milestone; the rationale and full
revisit plan stays in `PLAN.md` so readers can find the exact scope
that was deferred and why.

  * **Typed exception hierarchy** (`Emily.ShapeError`,
    `Emily.DtypeError`, `Emily.MLXError`). Re-evaluate at the 2.x
    line. See PLAN M19.
  * **GPU interop pointers** (`from_pointer` / `to_pointer` on
    `Nx.Backend`, plus a public `include/emily.h` for downstream
    NIFs). Revisit when a concrete downstream consumer asks. See
    PLAN M20.
  * **`mix emily.doctor` extensions for source-build diagnostics.**
    The Mix task itself shipped in 0.4.x for the precompiled-NIF
    path; the broader source-build probe set (Xcode CLT, CMake
    version skew, MLX source-tree state) is deferred until adoption
    surfaces a pattern of failures that `elixir_make` errors don't
    already explain. See PLAN M21.

## In-roadmap MLX capability gaps

Catalogued from the 2026-04-22 audit against MLX 0.31.1+69. Items
already shipped (`einsum`, SDPA sinks, microscaled quantization
modes) are recorded in PLAN. The remaining open items:

| #   | Capability                                                                | Status                | Trigger to revisit                                |
| --- | ------------------------------------------------------------------------- | --------------------- | ------------------------------------------------- |
| B3  | Sparse / MoE matmuls: `gather_qmm`, `gather_mm`, `block_masked_mm`, `segmented_mm` | Deferred              | First MoE model target (e.g. a Qwen3-MoE variant) |
| B4b | FP8 dtype (`to_fp8` / `from_fp8`)                                         | Blocked on Nx upstream | Nx gains FP8, or M16 surfaces a concrete user story |
| B5  | `ThreadLocalStream` / `set_default_stream`                                | Investigative         | Spike to confirm whether it simplifies the per-worker model |

A defn-callable fallback for `Emily.Fast.einsum/2` (currently
eager-only) is also open if a user surfaces cross-backend
composability needs — see PLAN M27.

## 1.0 release

Tracking checklist:

  * API docs and HexDocs reviewed for stale references — see issue #96.
  * `CHANGELOG.md` accumulated across releases (it is, since 0.3.0).
  * `MAINTAINING.md` reflects the precompiled-NIF release flow (it
    does, since 0.3.0).
  * Worked Bumblebee + quantized-Qwen3 examples in `notebooks/`
    (present and grouped in the HexDocs Notebooks section).
