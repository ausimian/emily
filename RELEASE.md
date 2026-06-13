### Fixed

- The README performance section now compares Emily against both
  benchmark baselines — EXLA (host CPU) and EMLX (the older MLX-backed
  Nx backend on the Metal GPU) — instead of EXLA alone, and its
  rule-of-thumb figures (ViT-base, DistilBERT) are reconciled with the
  current benchmark report.
- The benchmark report's environment block now records the Emily
  version the numbers were produced on (0.7.0) and drops a misleading
  run timestamp.
- The `MAINTAINING.md` release runbook is corrected: `mix publisho` is
  no longer described as pushing (it only commits and tags), and the
  obsolete manual draft-promotion step is dropped — `release-nif.yml`
  now publishes the release automatically once the NIFs are built.
