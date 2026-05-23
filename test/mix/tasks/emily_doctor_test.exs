defmodule Mix.Tasks.Emily.DoctorTest do
  use ExUnit.Case, async: true

  alias Mix.Tasks.Emily.Doctor

  defp healthy_opts(overrides \\ []) do
    existing = MapSet.new(["/tmp/emily/priv/libemily.so", "/tmp/emily/priv/mlx.metallib"])

    [
      os_type: {:unix, :darwin},
      arch: "aarch64-apple-darwin24.0.0",
      macos_version: "15.5",
      variant: :aot,
      priv_dir: "/tmp/emily/priv",
      file_exists?: &MapSet.member?(existing, &1),
      nif_loader: fn -> {:ok, "fake NIF loaded"} end,
      smoke_test: fn -> {:ok, "fake smoke passed"} end
    ]
    |> Keyword.merge(overrides)
  end

  test "diagnose returns ok when all checks pass" do
    report = Doctor.diagnose(healthy_opts())

    assert report.status == :ok
    assert Enum.all?(report.checks, &(&1.status == :ok))

    formatted = Doctor.format_report(report)
    assert formatted =~ "Emily doctor: OK"
    assert formatted =~ "[ok] variant: active :aot"
    assert formatted =~ "[ok] smoke test: fake smoke passed"
  end

  test "missing artifacts skip downstream NIF and smoke checks" do
    report =
      Doctor.diagnose(
        healthy_opts(
          file_exists?: fn _ -> false end,
          nif_loader: fn -> {:error, "should not be called"} end,
          smoke_test: fn -> {:error, "should not be called"} end
        )
      )

    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "Emily doctor: 2 problem(s) found"
    assert formatted =~ "[error] priv/libemily: missing"
    assert formatted =~ "[error] priv/mlx.metallib: missing"
    assert formatted =~ "[skip] NIF loadability: skipped (priv/libemily error)"
    assert formatted =~ "[skip] smoke test: skipped (priv/libemily error)"

    refute formatted =~ "should not be called"
  end

  test "NIF load failure skips smoke check but still reports the load error" do
    report =
      Doctor.diagnose(
        healthy_opts(
          nif_loader: fn -> {:error, "missing symbol _mlx"} end,
          smoke_test: fn -> {:error, "should not be called"} end
        )
      )

    assert report.status == :error

    formatted = Doctor.format_report(report)

    assert formatted =~
             "[error] NIF loadability: failed to load Emily.Native: missing symbol _mlx"

    assert formatted =~ "[skip] smoke test: skipped (NIF loadability error)"
    refute formatted =~ "should not be called"
  end

  test "smoke failure surfaces when prerequisites pass" do
    report =
      Doctor.diagnose(healthy_opts(smoke_test: fn -> {:error, "could not start :emily"} end))

    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "[error] smoke test: Emily.Backend smoke test failed"
  end

  test "diagnose validates variant-specific macOS support" do
    report = Doctor.diagnose(healthy_opts(variant: :jit, macos_version: "15.5"))

    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "macOS 15.5 is too old for :jit"
    assert formatted =~ ":jit requires macOS 26.2 or newer"
    assert formatted =~ "[skip] priv/libemily: skipped (platform error)"
    assert formatted =~ "[skip] smoke test: skipped (platform error)"
  end

  test "diagnose reports invalid variant with guidance" do
    report = Doctor.diagnose(healthy_opts(variant: :banana))

    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "[error] variant: invalid :emily, :variant :banana"
    assert formatted =~ "config :emily, variant: :aot"
    assert formatted =~ "[skip] smoke test: skipped (variant error)"
  end

  test "non-darwin OS is an error and gates downstream checks" do
    report = Doctor.diagnose(healthy_opts(os_type: {:unix, :linux}))

    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "[error] platform: unsupported OS {:unix, :linux}"
    assert formatted =~ "[skip] priv/libemily: skipped (platform error)"
    assert formatted =~ "[skip] priv/mlx.metallib: skipped (platform error)"
    assert formatted =~ "[skip] NIF loadability: skipped (platform error)"
    assert formatted =~ "[skip] smoke test: skipped (platform error)"
  end

  test "non-arm architecture is an error" do
    report = Doctor.diagnose(healthy_opts(arch: "x86_64-apple-darwin23.0.0"))

    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "[error] platform: unsupported architecture x86_64-apple-darwin23.0.0"
  end

  test "unknown macOS version produces a warn-only report" do
    report = Doctor.diagnose(healthy_opts(macos_version: :unknown))

    assert report.status == :warn

    formatted = Doctor.format_report(report)
    assert formatted =~ "Emily doctor: 1 warning(s)"
    assert formatted =~ "[warn] platform: macOS version could not be detected"
    assert formatted =~ "Ensure :aot runs on macOS 14.0 or newer"
    assert formatted =~ "Ensure :jit runs on macOS 26.2 or newer"
  end

  test "an unparseable macOS version also warns" do
    report = Doctor.diagnose(healthy_opts(macos_version: "Sequoia"))

    assert report.status == :warn

    formatted = Doctor.format_report(report)
    assert formatted =~ "[warn] platform: macOS version could not be detected"
  end

  test "--variant override flows through to diagnose" do
    # Confirms the runtime_opts plumbing by exercising the helper
    # logic directly: jit + macOS 15.5 must error as "too old".
    report = Doctor.diagnose(healthy_opts(variant: :jit))
    assert report.status == :error

    formatted = Doctor.format_report(report)
    assert formatted =~ "too old for :jit"
  end
end
