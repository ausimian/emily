### Fixed

- The README performance section now compares Emily against both
  benchmark baselines — EXLA (host CPU) and EMLX (the older MLX-backed
  Nx backend on the Metal GPU) — instead of EXLA alone, and its
  rule-of-thumb figures (ViT-base, DistilBERT) are reconciled with the
  current benchmark report.
- The benchmark report's environment block now records the Emily
  version the numbers were produced on (0.7.0) and drops a misleading
  run timestamp.
