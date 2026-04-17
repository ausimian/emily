defmodule Emily.MixedPrecisionTest do
  use ExUnit.Case, async: true

  alias Emily.MixedPrecision, as: MP
  alias Emily.MixedPrecision.LossScaler

  import Emily.BackendGenerators, only: [assert_close: 3]

  # -------------------------------------------------------------------
  # LossScaler.new/1
  # -------------------------------------------------------------------

  describe "LossScaler.new/1" do
    test "defaults" do
      s = LossScaler.new()
      assert s.scale == 65_536.0
      assert s.growth_factor == 2.0
      assert s.backoff_factor == 0.5
      assert s.growth_interval == 2000
      assert s.min_scale == 1.0
      assert s.counter == 0
    end

    test "custom opts" do
      s = LossScaler.new(scale: 1024.0, growth_interval: 500, min_scale: 0.5)
      assert s.scale == 1024.0
      assert s.growth_interval == 500
      assert s.min_scale == 0.5
    end
  end

  # -------------------------------------------------------------------
  # cast_params/2
  # -------------------------------------------------------------------

  describe "cast_params/2" do
    test "casts f32 tensors to bf16" do
      params = %{
        w: Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}),
        b: Nx.tensor([0.5], type: {:f, 32})
      }

      bf16 = MP.cast_params(params, {:bf, 16})
      assert bf16.w.type == {:bf, 16}
      assert bf16.b.type == {:bf, 16}
      assert_close(bf16.w, Nx.tensor([1.0, 2.0, 3.0], type: {:bf, 16}), tol: 1.0e-2)
    end

    test "casts bf16 tensors to f32" do
      params = %{w: Nx.tensor([1.0, 2.0], type: {:bf, 16})}
      f32 = MP.cast_params(params, {:f, 32})
      assert f32.w.type == {:f, 32}
    end

    test "leaves integer tensors unchanged" do
      params = %{
        w: Nx.tensor([1.0, 2.0], type: {:f, 32}),
        idx: Nx.tensor([0, 1, 2], type: {:s, 32})
      }

      bf16 = MP.cast_params(params, {:bf, 16})
      assert bf16.w.type == {:bf, 16}
      assert bf16.idx.type == {:s, 32}
    end

    test "handles nested maps" do
      params = %{
        layer1: %{w: Nx.tensor([1.0], type: {:f, 32}), b: Nx.tensor([0.0], type: {:f, 32})},
        layer2: %{w: Nx.tensor([2.0], type: {:f, 32})}
      }

      bf16 = MP.cast_params(params, {:bf, 16})
      assert bf16.layer1.w.type == {:bf, 16}
      assert bf16.layer1.b.type == {:bf, 16}
      assert bf16.layer2.w.type == {:bf, 16}
    end

    test "handles tuples" do
      params = {Nx.tensor([1.0], type: {:f, 32}), Nx.tensor([2.0], type: {:f, 32})}
      {a, b} = MP.cast_params(params, {:bf, 16})
      assert a.type == {:bf, 16}
      assert b.type == {:bf, 16}
    end

    test "handles lists" do
      params = [Nx.tensor([1.0], type: {:f, 32}), Nx.tensor([2.0], type: {:f, 32})]
      [a, b] = MP.cast_params(params, {:bf, 16})
      assert a.type == {:bf, 16}
      assert b.type == {:bf, 16}
    end

    test "passes through non-tensor values" do
      params = %{w: Nx.tensor([1.0], type: {:f, 32}), name: "layer1", count: 42}
      result = MP.cast_params(params, {:bf, 16})
      assert result.w.type == {:bf, 16}
      assert result.name == "layer1"
      assert result.count == 42
    end

    test "round-trip f32 -> bf16 -> f32 preserves values within bf16 tolerance" do
      original = Nx.tensor([0.1, 0.5, 1.0, -2.5, 3.14], type: {:f, 32})
      round_tripped = original |> MP.cast_params({:bf, 16}) |> MP.cast_params({:f, 32})
      assert_close(round_tripped, original, tol: 1.0e-2)
    end
  end

  # -------------------------------------------------------------------
  # accumulate_grad/2
  # -------------------------------------------------------------------

  describe "accumulate_grad/2" do
    test "produces same result as cast_params" do
      grads = %{w: Nx.tensor([0.01, -0.02], type: {:bf, 16})}
      assert MP.accumulate_grad(grads, {:f, 32}) == MP.cast_params(grads, {:f, 32})
    end
  end

  # -------------------------------------------------------------------
  # has_overflow?/1
  # -------------------------------------------------------------------

  describe "has_overflow?/1" do
    test "returns false for finite tensors" do
      grads = %{w: Nx.tensor([1.0, 2.0, 3.0]), b: Nx.tensor([0.5])}
      refute MP.has_overflow?(grads)
    end

    test "returns true when any tensor contains NaN" do
      grads = %{w: Nx.tensor([1.0, :nan, 3.0])}
      assert MP.has_overflow?(grads)
    end

    test "returns true when any tensor contains Inf" do
      grads = %{w: Nx.tensor([1.0, :infinity, 3.0])}
      assert MP.has_overflow?(grads)
    end

    test "returns true when any tensor contains -Inf" do
      grads = %{w: Nx.tensor([1.0, :neg_infinity, 3.0])}
      assert MP.has_overflow?(grads)
    end

    test "detects overflow in deeply nested structure" do
      grads = %{
        layer1: %{w: Nx.tensor([1.0, 2.0]), b: Nx.tensor([0.5])},
        layer2: %{w: Nx.tensor([1.0, :nan])}
      }

      assert MP.has_overflow?(grads)
    end

    test "returns false for integer tensors with large values" do
      grads = %{counts: Nx.tensor([999_999_999], type: {:s, 32})}
      refute MP.has_overflow?(grads)
    end

    test "returns false for empty map" do
      refute MP.has_overflow?(%{})
    end
  end

  # -------------------------------------------------------------------
  # scale_loss/2
  # -------------------------------------------------------------------

  describe "scale_loss/2" do
    test "multiplies loss by scaler.scale" do
      scaler = MP.loss_scale(scale: 1024.0)
      loss = Nx.tensor(0.5, type: {:f, 32})
      scaled = MP.scale_loss(loss, scaler)
      assert_close(scaled, Nx.tensor(512.0), tol: 1.0e-5)
    end

    test "works with bf16 loss" do
      scaler = MP.loss_scale(scale: 256.0)
      loss = Nx.tensor(0.25, type: {:bf, 16})
      scaled = MP.scale_loss(loss, scaler)
      assert_close(scaled, Nx.tensor(64.0, type: {:bf, 16}), tol: 1.0e-1)
    end
  end

  # -------------------------------------------------------------------
  # unscale/2
  # -------------------------------------------------------------------

  describe "unscale/2" do
    test "divides grads by scale, reports no overflow for finite grads" do
      scaler = MP.loss_scale(scale: 1024.0)
      grads = %{w: Nx.tensor([1024.0, 2048.0], type: {:f, 32})}
      {unscaled, overflow?} = MP.unscale(grads, scaler)

      refute overflow?
      assert_close(unscaled.w, Nx.tensor([1.0, 2.0]), tol: 1.0e-5)
    end

    test "detects NaN overflow" do
      scaler = MP.loss_scale()
      grads = %{w: Nx.tensor([1.0, :nan], type: {:f, 32})}
      {_unscaled, overflow?} = MP.unscale(grads, scaler)
      assert overflow?
    end

    test "detects Inf overflow" do
      scaler = MP.loss_scale()
      grads = %{w: Nx.tensor([:infinity, 1.0], type: {:f, 32})}
      {_unscaled, overflow?} = MP.unscale(grads, scaler)
      assert overflow?
    end

    test "leaves integer tensors unchanged" do
      scaler = MP.loss_scale(scale: 100.0)
      grads = %{w: Nx.tensor([100.0], type: {:f, 32}), idx: Nx.tensor([5], type: {:s, 32})}
      {unscaled, _overflow?} = MP.unscale(grads, scaler)
      assert unscaled.idx |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list() == [5]
    end
  end

  # -------------------------------------------------------------------
  # update/2
  # -------------------------------------------------------------------

  describe "update/2" do
    test "halves scale on overflow" do
      scaler = MP.loss_scale(scale: 1024.0)
      updated = MP.update(scaler, true)
      assert updated.scale == 512.0
      assert updated.counter == 0
    end

    test "floors at min_scale on overflow" do
      scaler = MP.loss_scale(scale: 1.0, min_scale: 1.0)
      updated = MP.update(scaler, true)
      assert updated.scale == 1.0
    end

    test "increments counter on success" do
      scaler = MP.loss_scale()
      updated = MP.update(scaler, false)
      assert updated.counter == 1
      assert updated.scale == scaler.scale
    end

    test "doubles scale after growth_interval steps" do
      scaler = MP.loss_scale(scale: 1024.0, growth_interval: 3)
      s1 = MP.update(scaler, false)
      s2 = MP.update(s1, false)
      s3 = MP.update(s2, false)

      assert s1.counter == 1
      assert s2.counter == 2
      assert s3.counter == 0
      assert s3.scale == 2048.0
    end

    test "resets counter on overflow" do
      scaler = %{MP.loss_scale(growth_interval: 10) | counter: 9}
      updated = MP.update(scaler, true)
      assert updated.counter == 0
    end

    test "resets counter on growth" do
      scaler = %{MP.loss_scale(growth_interval: 2) | counter: 1}
      updated = MP.update(scaler, false)
      assert updated.counter == 0
      assert updated.scale == 65_536.0 * 2.0
    end
  end

  # -------------------------------------------------------------------
  # loss_scale/1
  # -------------------------------------------------------------------

  describe "loss_scale/1" do
    test "returns a LossScaler with defaults" do
      s = MP.loss_scale()
      assert %LossScaler{} = s
      assert s.scale == 65_536.0
    end

    test "accepts custom options" do
      s = MP.loss_scale(scale: 512.0, growth_interval: 100)
      assert s.scale == 512.0
      assert s.growth_interval == 100
    end
  end

  # -------------------------------------------------------------------
  # Integration: deliberate bf16 overflow -> dynamic scaling stabilises
  # -------------------------------------------------------------------

  describe "dynamic loss scaling stabilisation" do
    test "scale halves on overflow and recovers" do
      scaler = MP.loss_scale(scale: 65_536.0, growth_interval: 3)

      # Simulate overflow: scale is too high for bf16.
      scaler = MP.update(scaler, true)
      assert scaler.scale == 32_768.0

      scaler = MP.update(scaler, true)
      assert scaler.scale == 16_384.0

      # Three successful steps trigger growth.
      scaler = MP.update(scaler, false)
      scaler = MP.update(scaler, false)
      scaler = MP.update(scaler, false)
      assert scaler.scale == 32_768.0
    end
  end
end
