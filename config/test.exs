import Config

# Production M22 flags stay default-false even in Emily's own tests so
# `mix test` doesn't pay GPU-sync cost on every run and we don't mask
# perf regressions. Fixture-only flags (set to `true`) drive the
# gate→helper composition tests in test/support/debug_fixture.ex.
config :emily,
  debug_bounds_check: false,
  debug_detect_nan_inf: false,
  test_fixture_debug_bounds_check: true,
  test_fixture_debug_detect_nan_inf: true

# Keep the native compiler strict in the test suite: an op the Expr
# compiler can't lower raises rather than silently falling back to the
# Evaluator. This preserves the no-fallback conformance gates (CM5) and
# the unsupported-op assertions. The runtime default is `:eval` (graceful
# fallback); tests that exercise the fallback pass `native_fallback: :eval`
# per call.
config :emily, native_fallback: :raise
