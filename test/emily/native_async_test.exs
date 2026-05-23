defmodule Emily.NativeAsyncTest do
  use ExUnit.Case, async: true

  alias Emily.Native.Async

  test "argument errors preserve reason and append context" do
    ref = make_ref()

    send(self(), {ref, {:error, {:argument, "bad argument"}}})

    err =
      assert_raise ArgumentError, fn ->
        Async.call(ref, %{
          op: :take,
          stream: :test_worker,
          inputs: [a: %{shape: [2, 3], dtype: {:f, 32}}],
          options: [axis: 1]
        })
      end

    assert err.message =~ "bad argument"
    assert err.message =~ "op=take"
    assert err.message =~ "a: shape=[2, 3] dtype={:f, 32}"
    assert err.message =~ "options=[axis: 1]"
    assert err.message =~ "stream=:test_worker"
  end

  test "runtime errors preserve reason and append context" do
    ref = make_ref()

    send(self(), {ref, {:error, {:runtime, "mlx runtime failed"}}})

    err =
      assert_raise RuntimeError, fn ->
        Async.call(ref, %{
          op: :linalg_cholesky,
          inputs: [a: %{shape: [2, 2], dtype: {:f, 32}}],
          options: [upper: false]
        })
      end

    assert err.message =~ "mlx runtime failed"
    assert err.message =~ "op=linalg_cholesky"
    assert err.message =~ "a: shape=[2, 2] dtype={:f, 32}"
    assert err.message =~ "options=[upper: false]"
  end
end
