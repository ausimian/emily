# Micro-benchmark: fused quantized_matmul (mx::quantized_matmul) vs the
# current quantized_dense path (dequantize_defn + Nx.dot), on GPU.
# No Bumblebee / model download needed. Single-token (batch=1) decode-shaped.
#
#   mix run bench/qmm_microbench.exs
alias Emily.Quantization
alias Emily.Quantization.Layers
alias Emily.QuantizedWeight

Nx.default_backend(Emily.Backend)

native = [compiler: Emily.Compiler, native: true]

dtype = :bf16
group_size = 64
bits = 4
warmup = 100
iters = 2000

# Qwen3-0.6B-shaped projections. Weight is [out, in] (transpose: true, the
# from_dense default); activation is [1, in] (one decode token).
shapes = [
  {"q_proj  [2048,1024]", 2048, 1024},
  {"kv_proj [1024,1024]", 1024, 1024},
  {"o_proj  [1024,2048]", 1024, 2048},
  {"mlp_up  [3072,1024]", 3072, 1024},
  {"mlp_dn  [1024,3072]", 1024, 3072}
]

# Force a full worker sync on the result (native :sync already blocks on
# mx::eval, but realizing the bytes is belt-and-suspenders).
sync = fn t -> Nx.to_binary(t) end

time_fn = fn compiled, x, qw ->
  Enum.each(1..warmup, fn _ -> compiled.(x, qw) end)
  sync.(compiled.(x, qw))
  t0 = System.monotonic_time(:microsecond)
  Enum.each(1..iters, fn _ -> compiled.(x, qw) end)
  sync.(compiled.(x, qw))
  t1 = System.monotonic_time(:microsecond)
  iters / ((t1 - t0) / 1_000_000)
end

IO.puts("dtype=#{dtype} group_size=#{group_size} bits=#{bits} warmup=#{warmup} iters=#{iters}\n")
IO.puts("  before = old quantized_dense (dequantize_defn + Nx.dot)")
IO.puts("  after  = quantized_dense now (#197: fused mx::quantized_matmul)\n")

IO.puts(String.pad_trailing("shape", 22) <> "  before(it/s)  after(it/s)  speedup  maxΔ")
IO.puts(String.duplicate("-", 68))

for {label, out_f, in_f} <- shapes do
  {w, _} = Nx.Random.normal(Nx.Random.key(0), shape: {out_f, in_f}, type: dtype)
  qw = QuantizedWeight.from_dense(w, group_size: group_size, bits: bits)
  {x, _} = Nx.Random.normal(Nx.Random.key(1), shape: {1, in_f}, type: dtype)

  # Pass the QuantizedWeight (an Nx.Container) as a jit ARGUMENT, not a
  # closure — its tensors become Expr params and its keep-metadata
  # (group_size/bits/transpose/mode) stays available at trace time. This
  # mirrors how Bumblebee threads quantized model params into the forward.
  #
  # `before` = the old layer body (dequantize the full bf16 weight, then
  # dense Nx.dot). `after` = the shipped layer, which now lowers to the
  # fused mx::quantized_matmul kernel.
  before = fn x, qw -> Nx.dot(x, Nx.transpose(Quantization.dequantize_defn(qw))) end
  after_fn = fn x, qw -> Layers.quantized_dense(x, qw) end

  before_compiled = Nx.Defn.jit(before, native)
  after_compiled = Nx.Defn.jit(after_fn, native)

  # correctness: after (fused) vs before (dequant), same math up to fp reorder
  b = before_compiled.(x, qw)
  a = after_compiled.(x, qw)
  max_delta = Nx.subtract(b, a) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  before_rate = time_fn.(before_compiled, x, qw)
  after_rate = time_fn.(after_compiled, x, qw)

  IO.puts(
    String.pad_trailing(label, 22) <>
      "  " <>
      String.pad_trailing(:erlang.float_to_binary(before_rate, decimals: 0), 12) <>
      "  " <>
      String.pad_trailing(:erlang.float_to_binary(after_rate, decimals: 0), 11) <>
      "  " <>
      String.pad_trailing(
        :erlang.float_to_binary(after_rate / before_rate, decimals: 2) <> "x",
        7
      ) <>
      "  " <> :erlang.float_to_binary(max_delta, decimals: 4)
  )
end
