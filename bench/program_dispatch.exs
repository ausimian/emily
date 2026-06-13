# CM0 dispatch-collapse microbench for the Expr->MLX single-NIF compiler.
#
# A chain of N `add`s built op-by-op costs N blocking BEAM<->worker NIF
# round-trips (the eager path the Evaluator walks today). The same chain
# compiled into one program and replayed via `eval_program` costs ONE
# round-trip — C++ rebuilds the whole mx::array DAG with no per-op BEAM
# hop. This bench measures that collapse.
#
# The lever the compiler pulls is the *build / dispatch* cost (the
# de-risk's "~68 ms of BEAM dispatch -> ~10 ms in C++"); the GPU floor is
# unchanged and shared by both paths. A dependent add chain forces N
# serial Metal kernels, so *total* wall time is GPU-floored and dilutes
# the ratio — therefore the gate is on build/dispatch time (eval_mode
# :build, no GPU), and total wall is reported separately for context.
#
#   mix run bench/program_dispatch.exs
#   mix run bench/program_dispatch.exs -- --chain 200 --iters 500
#
# The CM0 gate is a >=5x build/dispatch collapse; the script exits
# non-zero below it so CI can treat it as a check.

alias Emily.{IR, Native, Program}

{opts, _, _} =
  OptionParser.parse(System.argv(),
    strict: [chain: :integer, iters: :integer, width: :integer, gate: :float]
  )

chain = opts[:chain] || 100
iters = opts[:iters] || 300
width = opts[:width] || 16
gate = opts[:gate] || 5.0

worker = Emily.MlxStream.worker(Emily.MlxStream.Default)

f32 = fn list, shape ->
  bin = for x <- list, into: <<>>, do: <<x * 1.0::float-32-native>>
  Native.from_binary(bin, shape, {:f, 32})
end

input = f32.(List.duplicate(1.0, width), [width])
bias = f32.(List.duplicate(1.0, width), [width])

ir = %IR{
  n_inputs: 1,
  captures: [bias],
  instrs:
    for k <- 0..(chain - 1) do
      left = if k == 0, do: {:input, 0}, else: {:instr, k - 1}
      %{opcode: :add, operands: [left, {:capture, 0}]}
    end,
  outputs: [{:instr, chain - 1}]
}

prog = Program.compile(ir)

# --- Build / dispatch only (the lever) ---
# Eager: issue `chain` lazy adds = `chain` BEAM<->worker round-trips.
eager_build = fn ->
  Enum.reduce(1..chain, input, fn _, a -> Native.add(worker, a, bias) end)
end

# Compiled: build the whole DAG in C++ in one round-trip, no eval.
compiled_build = fn -> Program.eval(worker, prog, [input], mode: :build) end

# --- Total wall (GPU-floored; both sync-eval the same graph) ---
eager_total = fn ->
  acc = Enum.reduce(1..chain, input, fn _, a -> Native.add(worker, a, bias) end)
  Native.eval(worker, acc)
end

compiled_total = fn ->
  [out] = Program.eval(worker, prog, [input], mode: :sync)
  Native.eval(worker, out)
end

# Correctness guard before timing — the two paths must agree.
[compiled_out] = Program.eval(worker, prog, [input], mode: :sync)
eager_out = Enum.reduce(1..chain, input, fn _, a -> Native.add(worker, a, bias) end)

unless Native.to_binary(worker, compiled_out) == Native.to_binary(worker, eager_out) do
  IO.puts(:stderr, "FAIL: compiled and eager outputs differ")
  System.halt(1)
end

warmup = fn f -> for _ <- 1..20, do: f.() end

time = fn f ->
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: f.() end)
  us / iters
end

for f <- [eager_build, compiled_build, eager_total, compiled_total], do: warmup.(f)

eager_build_us = time.(eager_build)
compiled_build_us = time.(compiled_build)
eager_total_us = time.(eager_total)
compiled_total_us = time.(compiled_total)

dispatch_ratio = eager_build_us / compiled_build_us
total_ratio = eager_total_us / compiled_total_us

IO.puts("""

Expr-compiler dispatch-collapse microbench
  chain length : #{chain} adds
  tensor width : #{width} f32
  iterations   : #{iters}

  build/dispatch (the lever, no GPU)
    eager    (#{chain} round-trips) : #{Float.round(eager_build_us, 1)} us/iter
    compiled (1 round-trip)        : #{Float.round(compiled_build_us, 1)} us/iter
    dispatch collapse              : #{Float.round(dispatch_ratio, 2)}x   (gate >= #{gate}x)

  total wall (GPU-floored: #{chain} serial kernels, shared by both)
    eager                          : #{Float.round(eager_total_us, 1)} us/iter
    compiled                       : #{Float.round(compiled_total_us, 1)} us/iter
    speedup                        : #{Float.round(total_ratio, 2)}x
""")

if dispatch_ratio >= gate do
  IO.puts("PASS")
else
  IO.puts(:stderr, "FAIL: dispatch collapse #{Float.round(dispatch_ratio, 2)}x below gate #{gate}x")
  System.halt(1)
end
