# CM6 mx::compile secondary-win microbench.
#
# The single-NIF replay (CM0) already collapses the ~750 per-op
# BEAM<->worker round-trips into one. CM6 adds an *opt-in* `:compiled`
# eval mode that wraps that replay in `mx::compile` (cached per stream,
# keyed off the Program). `mx::compile` can fuse adjacent elementwise
# kernels, so the question this bench answers is narrow: on a realistic
# transformer block, does the compiled wrap buy a meaningful secondary
# encode win over the already-collapsed sync replay?
#
# The prior (Emily's own M6 milestone) predicted ~1.04-1.11x: MLX does
# not fuse matmul with adjacent elementwise. In practice this bench
# measures ~1.6x at decode-shaped sizes -- because at small sequence
# lengths kernel-launch and intermediate-memory overhead dominate, and
# fusing the elementwise runs (rms-norm, softmax, SiLU gating, residuals)
# removes them. The speedup is reported, not gated (only the correctness
# guard halts) -- CM6 is additive, not load-bearing: CM0's single-NIF
# replay already delivered the main dispatch collapse, and CM3 met the
# tok/s target without this.
#
#   mix run bench/program_compile.exs
#   mix run bench/program_compile.exs -- --seq 128 --iters 300
#
# Both paths run the SAME compiled Program; only the eval mode differs,
# and a correctness guard asserts they agree within a tight f32 tolerance
# before timing (mx::compile fuses elementwise runs, which reassociates
# float arithmetic to within a few ULP -- correct, not bit-exact).

alias Emily.{IR, Native, Program}
alias Nx.Defn.Composite

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [seq: :integer, dmodel: :integer, ffn: :integer, iters: :integer]
  )

seq = opts[:seq] || 64
d = opts[:dmodel] || 512
ffn = opts[:ffn] || 2048
iters = opts[:iters] || 300

worker = Emily.MlxStream.default_worker()

# --- A realistic transformer block, in Nx primitives only ---
# RMSNorm -> single-head attention (QKV matmuls, softmax, out proj) ->
# residual -> RMSNorm -> SwiGLU FFN -> residual. The elementwise runs
# (rms-norm, softmax, SiLU gating, residual adds) are mx::compile's
# fusion surface; the matmuls are the floor it can't fuse through.
eps = 1.0e-6
scale = 1.0 / :math.sqrt(d)

rms = fn t, g ->
  ms = Nx.mean(Nx.multiply(t, t), axes: [-1], keep_axes: true)
  Nx.multiply(Nx.divide(t, Nx.sqrt(Nx.add(ms, eps))), g)
end

silu = fn z -> Nx.multiply(z, Nx.divide(1.0, Nx.add(1.0, Nx.exp(Nx.negate(z))))) end

softmax = fn s ->
  m = Nx.reduce_max(s, axes: [-1], keep_axes: true)
  e = Nx.exp(Nx.subtract(s, m))
  Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
end

# Param order: x, g1, w_q, w_k, w_v, w_o, g2, w_gate, w_up, w_down.
block = fn [x, g1, w_q, w_k, w_v, w_o, g2, w_gate, w_up, w_down] ->
  h = rms.(x, g1)
  q = Nx.dot(h, w_q)
  k = Nx.dot(h, w_k)
  v = Nx.dot(h, w_v)
  scores = Nx.multiply(Nx.dot(q, Nx.transpose(k)), scale)
  ctx = Nx.dot(softmax.(scores), v)
  x2 = Nx.add(x, Nx.dot(ctx, w_o))

  f = rms.(x2, g2)
  gated = Nx.multiply(silu.(Nx.dot(f, w_gate)), Nx.dot(f, w_up))
  Nx.add(x2, Nx.dot(gated, w_down))
end

# Concrete weights on Emily.Backend; the param exprs become {:input, i}.
dev = fn shape ->
  dims = Tuple.to_list(shape)
  n = Enum.product(dims)
  bin = for i <- 1..n, into: <<>>, do: <<:math.sin(i * 0.013) * 0.1::float-32-native>>
  Native.from_binary(bin, dims, {:f, 32})
end

shapes = [
  {seq, d},
  {d},
  {d, d},
  {d, d},
  {d, d},
  {d, d},
  {d},
  {d, ffn},
  {d, ffn},
  {ffn, d}
]

input_refs = Enum.map(shapes, dev)

# Trace the block into an Expr (params at :root) and lower it once.
vars =
  shapes
  |> Enum.with_index()
  |> Enum.map(fn {shape, i} ->
    Nx.Defn.Expr.parameter(Nx.template(shape, {:f, 32}), :root, i)
  end)

expr = block.(vars)
{_template, leaves_rev} = Composite.traverse(expr, [], fn leaf, acc -> {leaf, [leaf | acc]} end)
prog = leaves_rev |> Enum.reverse() |> IR.lower() |> Program.compile()

# Correctness guard: compiled wrap must match the plain replay. On a
# shallow / integer-stable program the two are bit-identical, but
# mx::compile fuses elementwise runs (rms-norm, the `* scale`, softmax,
# SiLU gating), which reassociates f32 arithmetic. Through a deep block
# that shows up as a last-few-ULP drift -- correct, just not bit-exact.
# So assert a tight absolute tolerance and report the actual drift.
tol = 1.0e-5
[sync_out] = Program.eval(worker, prog, input_refs, mode: :sync)
[compiled_out] = Program.eval(worker, prog, input_refs, mode: :compiled)

floats = fn ref ->
  for <<v::float-32-native <- Native.to_binary(worker, ref)>>, do: v
end

sync_floats = floats.(sync_out)
compiled_floats = floats.(compiled_out)

# Guard length first: Enum.zip/2 truncates to the shorter list, so a real
# shape/length divergence between the two modes would otherwise be hidden.
if length(sync_floats) != length(compiled_floats) do
  IO.puts(:stderr, "FAIL: compiled output length differs from sync output length")
  System.halt(1)
end

max_drift =
  Enum.zip(sync_floats, compiled_floats)
  |> Enum.reduce(0.0, fn {a, b}, acc -> max(acc, abs(a - b)) end)

if max_drift > tol do
  IO.puts(:stderr, "FAIL: compiled vs sync max drift #{max_drift} exceeds tol #{tol}")
  System.halt(1)
end

build = fn -> Program.eval(worker, prog, input_refs, mode: :build) end

sync = fn ->
  [out] = Program.eval(worker, prog, input_refs, mode: :sync)
  Native.eval(worker, out)
end

compiled = fn ->
  [out] = Program.eval(worker, prog, input_refs, mode: :compiled)
  Native.eval(worker, out)
end

warmup = fn f -> for _ <- 1..30, do: f.() end
time = fn f ->
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: f.() end)
  us / iters
end

for f <- [build, sync, compiled], do: warmup.(f)

build_us = time.(build)
sync_us = time.(sync)
compiled_us = time.(compiled)

IO.puts("""

CM6 mx::compile secondary-win microbench
  transformer block : seq=#{seq} d_model=#{d} ffn=#{ffn}
  iterations        : #{iters}

  build only (dispatch, no GPU)  : #{Float.round(build_us, 1)} us/iter
  sync replay (eval)             : #{Float.round(sync_us, 1)} us/iter
  compiled replay (mx::compile)  : #{Float.round(compiled_us, 1)} us/iter

  compiled / sync speedup        : #{Float.round(sync_us / compiled_us, 3)}x
  compiled vs sync max drift     : #{max_drift} (f32 fusion reassociation, tol #{tol})
""")

cond do
  sync_us / compiled_us >= 1.09 ->
    IO.puts("RESULT: meets the ~1.09x M6 prediction -- compiled mode earns its keep.")

  sync_us / compiled_us >= 1.05 ->
    IO.puts("RESULT: marginal (1.05-1.09x). Opt-in is justified; stays off by default.")

  true ->
    IO.puts(
      "RESULT: below 1.05x. As the plan anticipated, mx::compile is not load-bearing; " <>
        ":compiled stays strictly opt-in and the single-NIF replay (CM0) is the real win."
    )
end
