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
