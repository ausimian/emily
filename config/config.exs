import Config

# Which MLX prebuilt Emily downloads at compile time. `:no_jit` (default)
# pulls the AOT-compiled metallib (~175 MB, works on older macOS);
# `:jit` pulls the JIT metallib (~25 MB, kernels compile on first use).
# The `@mlx_checksums` map in mix.exs is keyed by the resulting asset
# name, so valid values are only `:no_jit` and `:jit`.
config :emily, mlx_variant: :no_jit

if File.exists?(Path.join(__DIR__, "#{config_env()}.exs")) do
  import_config "#{config_env()}.exs"
end

# Optional per-checkout override (gitignored). Used by CI to flip the
# variant for the JIT lane without a custom MIX_ENV. Must stay after
# the env-file import so it takes final precedence.
if File.exists?(Path.join(__DIR__, "local.exs")) do
  import_config "local.exs"
end
