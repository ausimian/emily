defmodule Emily.MemoryTest do
  use ExUnit.Case, async: false
  doctest Emily.Memory

  alias Emily.Memory

  def forward_to_test(event, measurements, metadata, %{pid: pid}) do
    send(pid, {:event, event, measurements, metadata})
  end

  describe "stats/0" do
    test "returns active/peak/cache measurements and emits telemetry" do
      event = [:emily, :memory, :stats]
      handler_id = "memory-stats-#{inspect(self())}"

      :ok =
        :telemetry.attach(
          handler_id,
          event,
          &__MODULE__.forward_to_test/4,
          %{pid: self()}
        )

      on_exit(fn -> :telemetry.detach(handler_id) end)

      result = Memory.stats()

      assert %{active: active, peak: peak, cache: cache} = result
      assert is_integer(active) and active >= 0
      assert is_integer(peak) and peak >= 0
      assert is_integer(cache) and cache >= 0

      assert_receive {:event, ^event, ^result, %{}}
    end
  end

  describe "allocator controls" do
    test "reset_peak/0 and clear_cache/0 return :ok" do
      assert :ok = Memory.reset_peak()
      assert :ok = Memory.clear_cache()
    end
  end
end
