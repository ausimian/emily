import Config

# Which MLX build variant Emily ships. `:aot` (default) uses the
# AOT-compiled metallib (~175 MB, works on older macOS); `:jit` uses
# the JIT metallib (~25 MB, kernels compile on first use). In-repo
# builds take the value from the `EMILY_MLX_VARIANT` env var so CI can
# flip lanes without committing an override; hex consumers should set
# `config :emily, :variant, :jit` in their own `config/config.exs`.
variant =
  case System.get_env("EMILY_MLX_VARIANT") do
    nil -> :aot
    "aot" -> :aot
    "jit" -> :jit
    other -> raise ~s|EMILY_MLX_VARIANT must be "aot" or "jit", got: #{inspect(other)}|
  end

config :emily, variant: variant

if File.exists?(Path.join(__DIR__, "#{config_env()}.exs")) do
  import_config "#{config_env()}.exs"
end
