### Fixed

- MLX prebuilt download now runs on a peer VM (`:peer.start_link/1` with
  stdio connection) so it is unaffected by Mix's code-path pruning
  during dep compilation. Previous releases crashed in the tagged
  `smoke-test` CI lane with `{:error, :nofile}` / "module :public_key
  is not available" on clean caches, because Mix removed the
  `:ssl`/`:public_key`/`:asn1`/`:inets` ebin directories from the
  parent VM's code path even though the apps were started. The peer
  node has a fresh code path, so standard `httpc` + `public_key` work
  without further shimming.
