### Fixed

- Installing emily from Livebook (`Mix.install`) no longer crashes while
  downloading the precompiled NIF with "Can't set long node name!". The
  download ran on a named peer VM, which fails to boot under a
  longnames-distributed parent when the hostname is not fully qualified.
  The peer now starts without distribution.
