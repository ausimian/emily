defmodule Emily.TelemetryTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  alias Emily.Telemetry

  setup do
    # Make sure dedup state doesn't bleed between tests.
    Telemetry.reset_dedup()

    on_exit(fn ->
      # Clean up any handlers the test attached; each test's handler
      # id ends with its own pid so detach-all is safe.
      :telemetry.list_handlers([])
      |> Enum.each(&:telemetry.detach(&1.id))
    end)

    :ok
  end

  # Module-captured handler to avoid telemetry's local-fn perf warning.
  def forward_to_test(event, measurements, metadata, %{pid: pid}) do
    send(pid, {:event, event, measurements, metadata})
  end

  defp attach(event, id) do
    :telemetry.attach(id, event, &__MODULE__.forward_to_test/4, %{pid: self()})
  end

  describe "[:emily, :fallback, *]" do
    test "fires start + stop with op and shape metadata on a fallback path" do
      attach([:emily, :fallback, :start], "fallback-start-#{inspect(self())}")
      attach([:emily, :fallback, :stop], "fallback-stop-#{inspect(self())}")

      t = Nx.tensor([1, 2, 3, 4], backend: Emily.Backend)
      acc = Nx.tensor(0, backend: Emily.Backend)
      _ = Nx.reduce(t, acc, fn a, b -> Nx.add(a, b) end)

      assert_receive {:event, [:emily, :fallback, :start], _,
                      %{op: :reduce, input_shapes: shapes}}

      assert shapes == [{4}, {}]

      assert_receive {:event, [:emily, :fallback, :stop], %{duration: dur},
                      %{op: :reduce, input_shapes: ^shapes}}

      assert is_integer(dur) and dur > 0
    end

    test "warns once per {op, input_shapes} over 100 calls when enabled" do
      Application.put_env(:emily, :warn_on_fallback, true)
      on_exit(fn -> Application.delete_env(:emily, :warn_on_fallback) end)

      t = Nx.tensor([1, 2, 3], backend: Emily.Backend)
      acc = Nx.tensor(0, backend: Emily.Backend)

      log =
        capture_log(fn ->
          for _ <- 1..100 do
            Nx.reduce(t, acc, fn a, b -> Nx.add(a, b) end)
          end
        end)

      occurrences =
        log |> String.split("\n") |> Enum.count(&(&1 =~ "fell back to Nx.BinaryBackend"))

      assert occurrences == 1
      assert log =~ "reduce"
    end

    test "warn_on_fallback is off by default" do
      # No Application.put_env — default should be silent.
      t = Nx.tensor([1, 2, 3], backend: Emily.Backend)
      acc = Nx.tensor(0, backend: Emily.Backend)

      log =
        capture_log(fn ->
          Nx.reduce(t, acc, fn a, b -> Nx.add(a, b) end)
        end)

      refute log =~ "fell back"
    end
  end

  describe "[:emily, :to_binary, *]" do
    test "fires on Emily.Backend.to_binary/2 (the Nx.to_binary path)" do
      attach([:emily, :to_binary, :stop], "to-bin-stop-#{inspect(self())}")

      t = Nx.tensor([1.0, 2.0, 3.0], backend: Emily.Backend)
      _ = Nx.to_binary(t)

      assert_receive {:event, [:emily, :to_binary, :stop], %{duration: dur},
                      %{shape: shape, dtype: dtype, byte_size: bs}}

      assert shape == {3}
      assert dtype == {:f, 32}
      assert bs == 12
      assert is_integer(dur) and dur >= 0
    end
  end

  describe "[:emily, :memory, :stats]" do
    test "memory_stats/0 emits active/peak/cache measurements" do
      attach([:emily, :memory, :stats], "mem-stats-#{inspect(self())}")

      result = Telemetry.memory_stats()

      assert %{active: a, peak: p, cache: c} = result
      assert is_integer(a) and a >= 0
      assert is_integer(p) and p >= 0
      assert is_integer(c) and c >= 0

      assert_receive {:event, [:emily, :memory, :stats], ^result, _}
    end
  end
end
