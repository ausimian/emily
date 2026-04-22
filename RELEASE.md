### Fixed

- **`mix compile` crash on a cold MLX download in a clean consumer
  project.** `http_download!/2` in `mix.exs` called
  `:public_key.cacerts_get/0` right after
  `Application.ensure_all_started(:ssl)`. The app-start path pulled
  `:public_key` in transitively, but the module itself was not
  guaranteed to be loaded at call time — the tag-triggered Hex
  smoke test on CI blew up with
  `UndefinedFunctionError ... module :public_key is not available`
  on 0.2.0. `http_download!` now force-loads the module via
  `:code.ensure_loaded/1` before touching it. Any checkout with a
  populated `~/Library/Caches/emily/mlx-<v>-*` directory skipped
  this path, which is why the break only surfaced in the first
  clean CI run.
