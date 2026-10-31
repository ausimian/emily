### Changed

- emily now requires Nx 1.0, and Axon 0.9 / Bumblebee 0.8 for the optional
  Axon and Bumblebee integrations. To stay on Nx 0.12, pin
  `{:emily, "~> 1.3.0"}`.
- The native compiler (`native: true`) now compiles `Nx.block`s that
  `Emily.Backend` has no native kernel for, including every Axon 0.9
  activation (GeLU, softmax, ReLU, SiLU and the rest). It lowers the block's
  default implementation, which is exactly what the Evaluator runs, so the
  results are identical. Without this, every Axon or Bumblebee model would
  fall back to the Evaluator under Axon 0.9.

### Fixed

- Under `native: true`, a hook passed in the `:hooks` option (for example
  by `Nx.Serving` when streaming Bumblebee text generation) was silently
  skipped: the defn compiled natively and the hook never ran. A defn with
  a hooked `io_call` now falls back to the Evaluator, which runs the hook.
  Named `io_call`s with no hook registered still compile natively.
