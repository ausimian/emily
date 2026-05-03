### Fixed

- `Nx.LinAlg.svd(tensor, full_matrices?: false)` on rank-2 inputs no
  longer routes through MLX's full-matrices SVD and post-slices —
  MLX's SVD has no thin switch, so the old path materialised the full
  m × m U on device and instantly OOM'd Metal for tall matrices like
  the Qwen3-0.6B embedder kernel (151936 × 1024 → ~92 GB U). The thin
  case now computes `G = MᵀM → eigh → S, V; U = MV / S` (or the
  symmetric `MMᵀ` route for wide matrices), keeping the decomposition
  at min(m, n)². See the `Emily.Backend` moduledoc Divergences section
  for the numerical caveat (the Gram step squares M's condition
  number). Refs #84.
- `mix docs` runs cleanly. The MNIST notebook referenced
  `Axon.Loop.trainer/2` (no such arity); three other inline references
  resolved to `@doc false` callees in upstream libraries
  (`Nx.Defn.Expr.optional/3`, `Bumblebee.Layers.rms_norm/2`) and
  triggered autolinker warnings on every doc build. The notebook now
  uses the correct `trainer/3` arity, and the prose references have
  been reshaped so the autolinker no longer follows them, keeping the
  build warning-free for future `--warnings-as-errors` enforcement.
  Refs #83.
