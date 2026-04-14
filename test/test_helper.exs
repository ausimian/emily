# MLX's Metal runtime isn't safe for concurrent kernel dispatch from
# multiple OS threads — six test modules hammering MLX in parallel
# SIGSEGVs the Metal driver on macOS 14 (and produces sporadic
# "A command encoder is already encoding" assertions on other
# versions). Serialise module execution unconditionally; the entire
# suite still completes in seconds. See
# `test/soak/backend_concurrency_test.exs` for the full story.
ExUnit.start(max_cases: 1)
