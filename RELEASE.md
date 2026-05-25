### Added

- Distributed collectives over MLX's communication backends
  (`Emily.Distributed`): `all_sum/2`, `all_max/2`, `all_min/2`,
  `all_gather/2`, `sum_scatter/2`, plus `send/3`/`recv/4` and group
  `init/1`/`rank/1`/`size/1`. Backed by `ring` (TCP) and `jaccl`
  (RDMA-over-Thunderbolt) backends.
- `Emily.Distributed.Launcher` for running multiple ranks as local BEAM
  peer nodes (the `:peer`-based equivalent of `mlx.launch --backend
  ring`), enabling single-machine development of distributed code.
