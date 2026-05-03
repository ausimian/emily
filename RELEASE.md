### Fixed

- `Emily.Compiler` now silently drops options it doesn't recognise
  instead of raising `ArgumentError`. This matches the behaviour of
  `Nx.Defn.Evaluator` and EXLA, and restores compatibility with
  higher-level libraries that forward caller-supplied options through
  the JIT compiler — notably `Axon.build/2`, whose contract states
  that "all other options are forwarded to the underlying JIT
  compiler". Hit when running a Bumblebee-built Axon model with
  `Axon.predict(..., global_layer_options: [output_hidden_states:
  true])` under Emily as the global defn compiler. Refs #81.
