# scripts/expr_op_coverage.exs
#
# Probe every Nx op through `Emily.Compiler, native: true,
# native_fallback: :raise` and report which ones don't lower yet.
#
# Each probe builds a tiny defn over a representative input. An op the
# IR lowers returns a tensor; an op it doesn't raises `ArgumentError`,
# whose message names what's missing ("does not yet lower op :foo" /
# "does not yet lower the block Foo"). Composite Nx ops that decompose
# to lower-level primitives (e.g. `Nx.square` -> `multiply`) are still
# reported by their top-level name — they pass when every primitive in
# their expansion lowers.
#
# Run:
#   mix run scripts/expr_op_coverage.exs
#
# Output:
#   - per-op line, sorted into OK / MISS / ERROR / UNSUPPORTED groups
#   - a final markdown checklist suitable for pasting into a GitHub issue

defmodule ExprOpCoverage do
  @opts [compiler: Emily.Compiler, native: true, native_fallback: :raise]

  def probe(name, fun, args) do
    jit = Nx.Defn.jit(fun, @opts)
    _ = apply(jit, args)
    {:ok, name}
  rescue
    e in ArgumentError ->
      msg = Exception.message(e)

      cond do
        msg =~ "does not yet lower op" -> {:miss, name, extract_op(msg)}
        msg =~ "does not yet lower the block" -> {:miss, name, extract_block(msg)}
        msg =~ "cannot lower" -> {:unsup, name, msg |> first_line()}
        msg =~ "is not supported" -> {:miss, name, msg |> first_line()}
        msg =~ "no fallback" -> {:miss, name, msg |> first_line()}
        true -> {:error, name, msg |> first_line()}
      end

    e ->
      {:error, name, Exception.message(e) |> first_line()}
  end

  defp extract_op(msg) do
    case Regex.run(~r/does not yet lower op (:[a-z_0-9]+)/, msg) do
      [_, op] -> "op #{op}"
      _ -> first_line(msg)
    end
  end

  defp extract_block(msg) do
    case Regex.run(~r/does not yet lower the block ([A-Za-z0-9_.]+)/, msg) do
      [_, mod] -> "block #{mod}"
      _ -> first_line(msg)
    end
  end

  defp first_line(msg), do: msg |> String.split("\n") |> hd() |> String.slice(0, 120)
end

# ---------- probe definitions ----------

t = fn list, opts -> Nx.tensor(list, opts) end
f = fn list -> Nx.tensor(list) end
s32 = fn list -> Nx.tensor(list, type: :s32) end
u8 = fn list -> Nx.tensor(list, type: :u8) end
c = fn list -> Nx.tensor(Enum.map(list, &Complex.new(&1, 0.0))) end

probes =
  [
    # ============ Unary elementwise ============
    {:unary, :exp, fn x -> Nx.exp(x) end, [f.([1.0])]},
    {:unary, :expm1, fn x -> Nx.expm1(x) end, [f.([1.0])]},
    {:unary, :log, fn x -> Nx.log(x) end, [f.([1.0])]},
    {:unary, :log1p, fn x -> Nx.log1p(x) end, [f.([1.0])]},
    {:unary, :sigmoid, fn x -> Nx.sigmoid(x) end, [f.([1.0])]},
    {:unary, :cos, fn x -> Nx.cos(x) end, [f.([1.0])]},
    {:unary, :sin, fn x -> Nx.sin(x) end, [f.([1.0])]},
    {:unary, :tan, fn x -> Nx.tan(x) end, [f.([1.0])]},
    {:unary, :cosh, fn x -> Nx.cosh(x) end, [f.([1.0])]},
    {:unary, :sinh, fn x -> Nx.sinh(x) end, [f.([1.0])]},
    {:unary, :tanh, fn x -> Nx.tanh(x) end, [f.([1.0])]},
    {:unary, :acosh, fn x -> Nx.acosh(x) end, [f.([1.5])]},
    {:unary, :asinh, fn x -> Nx.asinh(x) end, [f.([1.0])]},
    {:unary, :atanh, fn x -> Nx.atanh(x) end, [f.([0.5])]},
    {:unary, :acos, fn x -> Nx.acos(x) end, [f.([0.5])]},
    {:unary, :asin, fn x -> Nx.asin(x) end, [f.([0.5])]},
    {:unary, :atan, fn x -> Nx.atan(x) end, [f.([0.5])]},
    {:unary, :sqrt, fn x -> Nx.sqrt(x) end, [f.([4.0])]},
    {:unary, :rsqrt, fn x -> Nx.rsqrt(x) end, [f.([4.0])]},
    {:unary, :cbrt, fn x -> Nx.cbrt(x) end, [f.([8.0])]},
    {:unary, :negate, fn x -> Nx.negate(x) end, [f.([1.0])]},
    {:unary, :sign, fn x -> Nx.sign(x) end, [f.([1.0])]},
    {:unary, :abs, fn x -> Nx.abs(x) end, [f.([-1.0])]},
    {:unary, :bitwise_not, fn x -> Nx.bitwise_not(x) end, [s32.([1])]},
    {:unary, :is_nan, fn x -> Nx.is_nan(x) end, [f.([1.0])]},
    {:unary, :is_infinity, fn x -> Nx.is_infinity(x) end, [f.([1.0])]},
    {:unary, :conjugate, fn x -> Nx.conjugate(x) end, [c.([1.0])]},
    {:unary, :real, fn x -> Nx.real(x) end, [c.([1.0])]},
    {:unary, :imag, fn x -> Nx.imag(x) end, [c.([1.0])]},
    {:unary, :floor, fn x -> Nx.floor(x) end, [f.([1.5])]},
    {:unary, :ceil, fn x -> Nx.ceil(x) end, [f.([1.5])]},
    {:unary, :round, fn x -> Nx.round(x) end, [f.([1.5])]},
    {:unary, :erf, fn x -> Nx.erf(x) end, [f.([1.0])]},
    {:unary, :erfc, fn x -> Nx.erfc(x) end, [f.([1.0])]},
    {:unary, :erf_inv, fn x -> Nx.erf_inv(x) end, [f.([0.5])]},
    {:unary, :bitcast, fn x -> Nx.bitcast(x, :s32) end, [f.([1.0])]},
    {:unary, :population_count, fn x -> Nx.population_count(x) end, [s32.([1])]},
    {:unary, :count_leading_zeros, fn x -> Nx.count_leading_zeros(x) end, [s32.([1])]},

    # ============ Binary arithmetic / bitwise ============
    {:binary, :add, fn a, b -> Nx.add(a, b) end, [f.([1.0]), f.([1.0])]},
    {:binary, :subtract, fn a, b -> Nx.subtract(a, b) end, [f.([1.0]), f.([1.0])]},
    {:binary, :multiply, fn a, b -> Nx.multiply(a, b) end, [f.([1.0]), f.([1.0])]},
    {:binary, :divide, fn a, b -> Nx.divide(a, b) end, [f.([1.0]), f.([1.0])]},
    {:binary, :pow, fn a, b -> Nx.pow(a, b) end, [f.([2.0]), f.([3.0])]},
    {:binary, :remainder, fn a, b -> Nx.remainder(a, b) end, [f.([5.0]), f.([3.0])]},
    {:binary, :atan2, fn a, b -> Nx.atan2(a, b) end, [f.([1.0]), f.([1.0])]},
    {:binary, :max, fn a, b -> Nx.max(a, b) end, [f.([1.0]), f.([2.0])]},
    {:binary, :min, fn a, b -> Nx.min(a, b) end, [f.([1.0]), f.([2.0])]},
    {:binary, :quotient, fn a, b -> Nx.quotient(a, b) end, [s32.([5]), s32.([2])]},
    {:binary, :bitwise_and, fn a, b -> Nx.bitwise_and(a, b) end, [s32.([1]), s32.([1])]},
    {:binary, :bitwise_or, fn a, b -> Nx.bitwise_or(a, b) end, [s32.([1]), s32.([1])]},
    {:binary, :bitwise_xor, fn a, b -> Nx.bitwise_xor(a, b) end, [s32.([1]), s32.([1])]},
    {:binary, :left_shift, fn a, b -> Nx.left_shift(a, b) end, [s32.([1]), s32.([2])]},
    {:binary, :right_shift, fn a, b -> Nx.right_shift(a, b) end, [s32.([4]), s32.([1])]},

    # ============ Compare / logical ============
    {:compare, :equal, fn a, b -> Nx.equal(a, b) end, [f.([1.0]), f.([1.0])]},
    {:compare, :not_equal, fn a, b -> Nx.not_equal(a, b) end, [f.([1.0]), f.([1.0])]},
    {:compare, :less, fn a, b -> Nx.less(a, b) end, [f.([1.0]), f.([2.0])]},
    {:compare, :less_equal, fn a, b -> Nx.less_equal(a, b) end, [f.([1.0]), f.([2.0])]},
    {:compare, :greater, fn a, b -> Nx.greater(a, b) end, [f.([1.0]), f.([2.0])]},
    {:compare, :greater_equal, fn a, b -> Nx.greater_equal(a, b) end, [f.([1.0]), f.([2.0])]},
    {:compare, :logical_and, fn a, b -> Nx.logical_and(a, b) end, [u8.([1]), u8.([0])]},
    {:compare, :logical_or, fn a, b -> Nx.logical_or(a, b) end, [u8.([1]), u8.([0])]},
    {:compare, :logical_xor, fn a, b -> Nx.logical_xor(a, b) end, [u8.([1]), u8.([0])]},
    {:compare, :logical_not, fn x -> Nx.logical_not(x) end, [u8.([0])]},

    # ============ Reductions ============
    {:reduce, :sum, fn x -> Nx.sum(x) end, [f.([1.0, 2.0])]},
    {:reduce, :product, fn x -> Nx.product(x) end, [f.([1.0, 2.0])]},
    {:reduce, :all, fn x -> Nx.all(x) end, [u8.([1])]},
    {:reduce, :any, fn x -> Nx.any(x) end, [u8.([1])]},
    {:reduce, :reduce_max, fn x -> Nx.reduce_max(x) end, [f.([1.0, 2.0])]},
    {:reduce, :reduce_min, fn x -> Nx.reduce_min(x) end, [f.([1.0, 2.0])]},
    {:reduce, :argmax, fn x -> Nx.argmax(x) end, [f.([1.0, 2.0])]},
    {:reduce, :argmin, fn x -> Nx.argmin(x) end, [f.([1.0, 2.0])]},

    # ============ Shape ops ============
    {:shape, :reshape, fn x -> Nx.reshape(x, {2, 1}) end, [f.([1.0, 2.0])]},
    {:shape, :squeeze, fn x -> Nx.squeeze(x, axes: [0]) end, [f.([[1.0]])]},
    {:shape, :transpose, fn x -> Nx.transpose(x) end, [f.([[1.0, 2.0]])]},
    {:shape, :as_type, fn x -> Nx.as_type(x, :s32) end, [f.([1.0])]},
    {:shape, :broadcast, fn x -> Nx.broadcast(x, {2, 2}) end, [f.([1.0, 2.0])]},
    {:shape, :pad, fn x -> Nx.pad(x, 0.0, [{1, 1, 0}]) end, [f.([1.0, 2.0])]},
    {:shape, :reverse, fn x -> Nx.reverse(x) end, [f.([1.0, 2.0])]},
    {:shape, :concatenate, fn a, b -> Nx.concatenate([a, b]) end, [f.([1.0]), f.([2.0])]},
    {:shape, :stack, fn a, b -> Nx.stack([a, b]) end, [f.([1.0]), f.([2.0])]},

    # ============ Linalg core ============
    {:linalg, :dot, fn a, b -> Nx.dot(a, b) end, [f.([1.0, 2.0]), f.([1.0, 2.0])]},
    {:linalg, :conv, fn x, k -> Nx.conv(x, k, strides: 1, padding: :valid) end,
     [f.([[[1.0, 2.0, 3.0]]]), f.([[[1.0]]])]},

    # ============ Selection / indexing ============
    {:select, :select, fn p, t, fa -> Nx.select(p, t, fa) end,
     [u8.([1]), f.([1.0]), f.([2.0])]},
    {:select, :clip, fn x -> Nx.clip(x, Nx.tensor(0.0), Nx.tensor(1.0)) end, [f.([0.5])]},
    {:select, :slice, fn x -> Nx.slice(x, [0], [1]) end, [f.([1.0, 2.0])]},
    {:select, :put_slice, fn x, u -> Nx.put_slice(x, [0], u) end, [f.([1.0, 2.0]), f.([3.0])]},
    {:select, :gather, fn x, i -> Nx.gather(x, i) end, [f.([1.0, 2.0]), s32.([[0], [1]])]},
    {:select, :indexed_put, fn x, i, u -> Nx.indexed_put(x, i, u) end,
     [f.([1.0, 2.0]), s32.([[0]]), f.([3.0])]},
    {:select, :indexed_add, fn x, i, u -> Nx.indexed_add(x, i, u) end,
     [f.([1.0, 2.0]), s32.([[0]]), f.([3.0])]},
    {:select, :take, fn x, i -> Nx.take(x, i) end, [f.([1.0, 2.0]), s32.([0])]},
    {:select, :take_along_axis, fn x, i -> Nx.take_along_axis(x, i) end,
     [f.([1.0, 2.0]), s32.([0])]},

    # ============ Sort / argsort / top_k ============
    {:sort, :sort, fn x -> Nx.sort(x) end, [f.([2.0, 1.0])]},
    {:sort, :argsort, fn x -> Nx.argsort(x) end, [f.([2.0, 1.0])]},
    {:sort, :top_k, fn x -> Nx.top_k(x, k: 1) end, [f.([2.0, 1.0])]},

    # ============ Creation ============
    {:create, :iota, fn -> Nx.iota({4}) end, []},
    {:create, :eye, fn -> Nx.eye(2) end, []},

    # ============ FFT family ============
    {:fft, :fft, fn x -> Nx.fft(x, length: 4) end, [f.([1.0, 0.0, 0.0, 0.0])]},
    {:fft, :ifft, fn x -> Nx.ifft(x, length: 4) end,
     [Nx.tensor([Complex.new(1.0, 0.0), Complex.new(0.0, 0.0), Complex.new(0.0, 0.0), Complex.new(0.0, 0.0)])]},
    {:fft, :fft2, fn x -> Nx.fft2(x) end, [Nx.iota({2, 2}, type: :f32)]},
    {:fft, :ifft2, fn x -> Nx.ifft2(x) end,
     [Nx.tensor([[Complex.new(1.0, 0.0), Complex.new(0.0, 0.0)], [Complex.new(0.0, 0.0), Complex.new(0.0, 0.0)]])]},
    {:fft, :rfft, fn x -> Nx.rfft(x, length: 4) end, [f.([1.0, 0.0, 0.0, 0.0])]},
    {:fft, :irfft, fn x -> Nx.irfft(x, length: 4) end,
     [Nx.tensor([Complex.new(1.0, 0.0), Complex.new(0.0, 0.0), Complex.new(0.0, 0.0)])]},

    # ============ Window (pooling) reductions ============
    {:window, :window_sum, fn x -> Nx.window_sum(x, {2}) end, [f.([1.0, 2.0, 3.0])]},
    {:window, :window_max, fn x -> Nx.window_max(x, {2}) end, [f.([1.0, 2.0, 3.0])]},
    {:window, :window_min, fn x -> Nx.window_min(x, {2}) end, [f.([1.0, 2.0, 3.0])]},
    {:window, :window_product, fn x -> Nx.window_product(x, {2}) end, [f.([1.0, 2.0, 3.0])]},
    {:window, :window_mean, fn x -> Nx.window_mean(x, {2}) end, [f.([1.0, 2.0, 3.0])]},

    # ============ Cumulative ============
    {:cumulative, :cumulative_sum, fn x -> Nx.cumulative_sum(x) end, [f.([1.0, 2.0])]},
    {:cumulative, :cumulative_product, fn x -> Nx.cumulative_product(x) end, [f.([1.0, 2.0])]},
    {:cumulative, :cumulative_max, fn x -> Nx.cumulative_max(x) end, [f.([1.0, 2.0])]},
    {:cumulative, :cumulative_min, fn x -> Nx.cumulative_min(x) end, [f.([1.0, 2.0])]},
    # interior-axis variants — the IR last-axis fast-path doesn't apply
    {:cumulative, :cumulative_sum_axis0, fn x -> Nx.cumulative_sum(x, axis: 0) end,
     [f.([[1.0, 2.0], [3.0, 4.0]])]},

    # ============ Linalg blocks ============
    {:linalg_block, :cholesky, fn x -> Nx.LinAlg.cholesky(x) end,
     [f.([[4.0, 2.0], [2.0, 3.0]])]},
    {:linalg_block, :svd, fn x -> Nx.LinAlg.svd(x) end, [f.([[1.0, 0.0], [0.0, 1.0]])]},
    {:linalg_block, :qr, fn x -> Nx.LinAlg.qr(x) end, [f.([[1.0, 0.0], [0.0, 1.0]])]},
    {:linalg_block, :eigh, fn x -> Nx.LinAlg.eigh(x) end, [f.([[2.0, 0.0], [0.0, 1.0]])]},
    {:linalg_block, :lu, fn x -> Nx.LinAlg.lu(x) end, [f.([[1.0, 0.0], [0.0, 1.0]])]},
    {:linalg_block, :determinant, fn x -> Nx.LinAlg.determinant(x) end,
     [f.([[1.0, 0.0], [0.0, 1.0]])]},
    {:linalg_block, :solve, fn a, b -> Nx.LinAlg.solve(a, b) end,
     [f.([[1.0, 0.0], [0.0, 1.0]]), f.([1.0, 1.0])]},
    {:linalg_block, :triangular_solve, fn a, b -> Nx.LinAlg.triangular_solve(a, b) end,
     [f.([[1.0, 0.0], [0.0, 1.0]]), f.([1.0, 1.0])]},

    # ============ Other Nx.Block ============
    {:block, :all_close, fn a, b -> Nx.all_close(a, b) end, [f.([1.0]), f.([1.0])]},
    {:block, :phase, fn x -> Nx.phase(x) end, [c.([1.0])]}
  ]

# ---------- run ----------

# Use the eager Emily.Backend so input materialization works.
Nx.global_default_backend(Emily.Backend)

results =
  Enum.map(probes, fn {cat, name, fun, args} ->
    {cat, name, ExprOpCoverage.probe({cat, name}, fun, args)}
  end)

# ---------- print plain log ----------

IO.puts("\n========= per-op result (#{length(results)} probes) =========\n")

grouped =
  Enum.reduce(results, %{ok: [], miss: [], unsup: [], error: []}, fn {cat, name, res}, acc ->
    tag =
      case res do
        {:ok, _} -> :ok
        {:miss, _, _} -> :miss
        {:unsup, _, _} -> :unsup
        {:error, _, _} -> :error
      end

    Map.update!(acc, tag, &[{cat, name, res} | &1])
  end)
  |> Map.new(fn {k, v} -> {k, Enum.reverse(v)} end)

for {tag, list} <- [{:ok, "OK"}, {:miss, "MISS"}, {:unsup, "UNSUPPORTED"}, {:error, "ERROR"}] do
  rows = Map.fetch!(grouped, tag)
  IO.puts("--- #{list} (#{length(rows)}) ---")

  for {cat, name, res} <- rows do
    detail =
      case res do
        {:ok, _} -> ""
        {_, _, why} -> " — #{why}"
      end

    IO.puts("  [#{cat}] #{name}#{detail}")
  end

  IO.puts("")
end

# ---------- print markdown checklist ----------

IO.puts("\n========= markdown checklist (paste into the issue) =========\n")

miss_by_cat =
  grouped.miss
  |> Enum.group_by(fn {cat, _, _} -> cat end)
  |> Enum.sort()

for {cat, rows} <- miss_by_cat do
  IO.puts("### #{cat} (#{length(rows)})")

  for {_, name, {_, _, why}} <- Enum.sort_by(rows, fn {_, n, _} -> n end) do
    IO.puts("- [ ] `#{name}` — #{why}")
  end

  IO.puts("")
end

unsup_rows = grouped.unsup

if unsup_rows != [] do
  IO.puts("### Unlowerable by design (#{length(unsup_rows)})")

  for {_, name, {_, _, why}} <- Enum.sort_by(unsup_rows, fn {_, n, _} -> n end) do
    IO.puts("- [ ] `#{name}` — #{why}")
  end

  IO.puts("")
end
