### Fixed

- Precompiled NIF download no longer times out on the `:peer.call/4`
  default 5s `gen_server.call` deadline. Consumers installing
  `{:emily, "~> 0.3"}` on a cold cache could see `:gen_server.call`
  timeouts while fetching the multi-MB tarball; the `.sha256` sidecar
  fit in the window but the main asset did not. The peer RPC now runs
  with `:infinity` so httpc's own request timing drives cancellation.
