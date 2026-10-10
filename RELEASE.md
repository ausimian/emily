### Fixed

- Compiling emily as a dependency no longer warns that
  `:public_key.cacerts_get/0` and
  `:public_key.pkix_verify_hostname_match_fun/1` are undefined. The
  warnings came from `mix emily.checksums`, a maintainer-only release
  task that shipped in the hex package. The package now leaves out the
  Mix tasks that only work in a checkout of the repository
  (`emily.checksums`, `bench.native` and `clang.tidy`); `mix emily.doctor`
  is still included.
