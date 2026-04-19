### Fixed

- **HexDocs source links.** `mix.exs`'s `source_url_pattern`
  prepended a `v` prefix to the version tag, but the project's
  release convention (via `mix publisho`) uses bare semver tags.
  The generated `[source]` links in HexDocs pointed at nonexistent
  `v<version>` tags. Dropped the prefix so links resolve to the
  actual tag.
