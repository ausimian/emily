# MLX's Metal runtime isn't safe for concurrent kernel dispatch from
# multiple OS threads — six test modules hammering MLX in parallel
# SIGSEGVs the Metal driver on macOS 14 (and produces sporadic
# "A command encoder is already encoding" assertions on other
# versions). Serialise module execution unconditionally; the entire
# suite still completes in seconds. See
# `test/soak/backend_concurrency_test.exs` for the full story.
#
# Conformance tests pull tiny-random HuggingFace models at runtime and
# take ~tens of seconds per model on a cold cache — opt-in via
# `mix test --only conformance`. `:qwen3_full` is an even heavier
# conformance variant that downloads ~1.5 GB of weights, so it is
# excluded even from `--only conformance`; run explicitly via
# `mix test --only qwen3_full`. (Soak tests deliberately stay in the
# default suite; see `test/soak/memory_test.exs` for the rationale.)
ExUnit.start(max_cases: 1, exclude: [:conformance, :qwen3_full])
