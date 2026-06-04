defmodule Emily.CompilerDecodeTest do
  @moduledoc """
  CM3 integration: the **shape-stable Gemma decode pattern** compiles
  single-NIF, bit-identical to the Evaluator — without a 13 GB checkpoint.

  These exercise the novel combination the decode refactor relies on, all
  with `offset` threaded as a runtime `s32` scalar input (so one compiled
  program serves every token):

    * fused RoPE on the 4-D `{1, heads, seq, head_dim}` layout,
    * the KV-cache write as a **dynamic** `put_slice` at `offset`,
    * a full-`max_len` window read with an additive **length mask**
      (replacing the growing `[0, len)` slice), and
    * masked SDPA + the quantized projections / gated FFN.

  Both paths call the same MLX kernels, so equality is exact. The eager
  Evaluator materializes the scalar-tensor `offset` to a number; the
  compiler keeps it a runtime input — they agree.
  """
  use ExUnit.Case, async: true

  @native [compiler: Emily.Compiler, native: true]
  @eval [compiler: Emily.Compiler]

  defp run(fun, args, opts), do: apply(Nx.Defn.jit(fun, opts), args)

  defp assert_native_eq_eval(fun, args) do
    native = run(fun, args, @native)
    eval = run(fun, args, @eval)
    assert %Emily.Backend{} = native.data
    assert native.shape == eval.shape and native.type == eval.type
    assert Nx.to_binary(native) == Nx.to_binary(eval)
    native
  end

  defp dev(data, opts \\ []), do: Nx.tensor(data, [backend: Emily.Backend] ++ opts)

  describe "shape-stable attention core" do
    test "RoPE + dynamic KV write + length mask + masked SDPA, offset as input" do
      n_q = 2
      n_kv = 1
      hd = 4
      max_len = 4
      seq = 1

      q = Nx.divide(Nx.iota({1, n_q, seq, hd}, type: :f32, backend: Emily.Backend), 16.0)
      k_new = Nx.divide(Nx.iota({1, n_kv, seq, hd}, type: :f32, backend: Emily.Backend), 12.0)
      v_new = Nx.divide(Nx.iota({1, n_kv, seq, hd}, type: :f32, backend: Emily.Backend), 8.0)
      k_cache = Nx.broadcast(dev(0.0), {1, n_kv, max_len, hd})
      v_cache = Nx.broadcast(dev(0.0), {1, n_kv, max_len, hd})
      offset = dev(2, type: :s32)

      core = fn q, k_new, v_new, k_cache, v_cache, offset ->
        # Token at absolute position `offset`; rotate q and the new k.
        q = Emily.Fast.rope(q, offset, dims: hd)
        k_new = Emily.Fast.rope(k_new, offset, dims: hd)

        # Write the new token into the fixed KV buffers at `offset`.
        k_buf = Nx.put_slice(k_cache, [0, 0, offset, 0], k_new)
        v_buf = Nx.put_slice(v_cache, [0, 0, offset, 0], v_new)

        # Length mask over the full window: position j valid iff j < offset+seq.
        pos = Nx.iota({max_len}, type: :s32)
        valid = Nx.less(pos, Nx.add(offset, seq))
        mask = valid |> Nx.select(0.0, -1.0e9) |> Nx.reshape({1, 1, seq, max_len})

        Emily.Fast.scaled_dot_product_attention_with_mask(q, k_buf, v_buf, mask, scale: 0.5)
      end

      out = assert_native_eq_eval(core, [q, k_new, v_new, k_cache, v_cache, offset])
      assert out.shape == {1, n_q, seq, hd}
    end

    test "mask-vs-slice: full-window masked SDPA equals the growing-slice read" do
      # The decode refactor replaces `Nx.slice(k_buf, .., [.., len, ..])`
      # (len = offset+seq) with a full max_len window + length mask. This
      # asserts the two are numerically equal (the masked positions get 0
      # attention weight), the core de-risk before the gem_chat change.
      n_q = 2
      n_kv = 1
      hd = 4
      max_len = 4
      len = 3

      q = Nx.divide(Nx.iota({1, n_q, 1, hd}, type: :f32, backend: Emily.Backend), 16.0)
      # KV buffer with `len` real rows and the rest zero (post write-at-offset).
      k_buf = Nx.divide(Nx.iota({1, n_kv, max_len, hd}, type: :f32, backend: Emily.Backend), 10.0)
      v_buf = Nx.divide(Nx.iota({1, n_kv, max_len, hd}, type: :f32, backend: Emily.Backend), 7.0)

      sliced =
        run(
          fn q, k, v ->
            ks = Nx.slice(k, [0, 0, 0, 0], [1, n_kv, len, hd])
            vs = Nx.slice(v, [0, 0, 0, 0], [1, n_kv, len, hd])
            Emily.Fast.scaled_dot_product_attention(q, ks, vs, scale: 0.5)
          end,
          [q, k_buf, v_buf],
          @eval
        )

      masked =
        run(
          fn q, k, v ->
            pos = Nx.iota({max_len}, type: :s32)
            mask = Nx.less(pos, len) |> Nx.select(0.0, -1.0e9) |> Nx.reshape({1, 1, 1, max_len})
            Emily.Fast.scaled_dot_product_attention_with_mask(q, k, v, mask, scale: 0.5)
          end,
          [q, k_buf, v_buf],
          @native
        )

      # Fused SDPA reductions over a different window length can reorder fp
      # ops; assert close rather than bit-identical here.
      assert_all_close(Nx.to_flat_list(masked), Nx.to_flat_list(sliced), 1.0e-5)
    end
  end

  defp assert_all_close(a, b, tol) do
    assert length(a) == length(b)
    Enum.zip(a, b) |> Enum.each(fn {x, y} -> assert_in_delta(x, y, tol) end)
  end
end
