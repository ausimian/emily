defmodule Emily.Fast.EinsumTest do
  @moduledoc """
  Tests for `Emily.Fast.einsum/2`.

  Every case runs on four lanes and is checked against a brute-force
  reference that loops over every label assignment in plain Elixir:

    * `eager` — a direct call on `Emily.Backend` tensors (`mx::einsum`).
    * `evaluator` — `defn` under `Emily.Compiler, native: false`, which
      dispatches the block to `mx::einsum` through `Emily.Backend`.
    * `native` — `defn` under `Emily.Compiler, native: true` with
      `native_fallback: :raise`, so the `einsum` IR opcode must lower.
    * `fallback` — `defn` under `Nx.Defn.Evaluator` on
      `Nx.BinaryBackend`, which runs the composed `Nx.dot/6` path.

  The reference takes the explicit, ellipsis-free form of each equation,
  written out by hand, so it does not share the implementation's parser.
  """

  use ExUnit.Case, async: true

  doctest Emily.Fast, only: [einsum: 2]

  import Emily.BackendGenerators, only: [assert_close: 3]

  alias Emily.FastFallback

  @f32_tol 1.0e-4

  defmodule Defns do
    @moduledoc false
    import Nx.Defn

    defn(attention_scores(q, k), do: Emily.Fast.einsum("bhid,bhjd->bhij", [q, k]) * 0.5)
  end

  # {equation, explicit equation for the reference, operand shapes}
  @cases [
    # contractions
    {"ij,jk->ik", "ij,jk->ik", [{3, 4}, {4, 5}]},
    {"bij,bjk->bik", "bij,bjk->bik", [{2, 3, 4}, {2, 4, 5}]},
    {"bhid,bhjd->bhij", "bhid,bhjd->bhij", [{2, 3, 4, 5}, {2, 3, 6, 5}]},
    {"i,i->", "i,i->", [{5}, {5}]},
    {"i,j->ij", "i,j->ij", [{3}, {4}]},
    {"ij,ij->ij", "ij,ij->ij", [{3, 4}, {3, 4}]},
    {"bij,jk->bki", "bij,jk->bki", [{2, 3, 4}, {4, 5}]},
    # multi-operand, including a label shared by non-adjacent operands
    {"ij,jk,kl->il", "ij,jk,kl->il", [{2, 3}, {3, 4}, {4, 5}]},
    {"ab,bc,cd,ad->", "ab,bc,cd,ad->", [{2, 3}, {3, 4}, {4, 2}, {2, 2}]},
    {"ab,bc,ca->b", "ab,bc,ca->b", [{2, 3}, {3, 4}, {4, 2}]},
    # single operand: transpose, reductions, diagonals
    {"ij->ji", "ij->ji", [{3, 4}]},
    {"ij->", "ij->", [{3, 4}]},
    {"ijk->kj", "ijk->kj", [{2, 3, 4}]},
    {"ii->i", "ii->i", [{4, 4}]},
    {"iii->i", "iii->i", [{3, 3, 3}]},
    {"iji->ij", "iji->ij", [{3, 2, 3}]},
    {"bii->bi", "bii->bi", [{2, 3, 3}]},
    {"ii,ij->j", "ii,ij->j", [{3, 3}, {3, 4}]},
    # implicit output: once-only labels, sorted, uppercase first
    {"ii", "ii->", [{4, 4}]},
    {"ji", "ji->ij", [{3, 4}]},
    {"ij,jk", "ij,jk->ik", [{3, 4}, {4, 5}]},
    {"bA,Ac", "bA,Ac->bc", [{2, 3}, {3, 4}]},
    {"Ab,bC->AC", "Ab,bC->AC", [{2, 3}, {3, 4}]},
    # ellipsis
    {"...ij,...jk->...ik", "abij,abjk->abik", [{2, 3, 4, 5}, {2, 3, 5, 6}]},
    {"...ij,...jk", "zij,zjk->zik", [{2, 3, 4}, {2, 4, 5}]},
    {"...ij,...jk->...ik", "yzij,zjk->yzik", [{2, 1, 3, 4}, {5, 4, 2}]},
    {"...ij,jk->...ik", "ij,jk->ik", [{3, 4}, {4, 5}]},
    {"i...->i", "iab->i", [{2, 3, 4}]},
    {"...i,...i->...", "zi,zi->z", [{3, 4}, {3, 4}]},
    {"b...d,b...d->b...", "bzd,bzd->bz", [{2, 3, 4}, {2, 3, 4}]},
    # size-1 broadcasting, scalars, spaces
    {"ij,ij->ij", "ij,ij->ij", [{3, 1}, {3, 4}]},
    {"ij,jk->ik", "ij,jk->ik", [{3, 4}, {1, 5}]},
    {",ij->ij", ",ij->ij", [{}, {2, 3}]},
    {" ij , jk -> ik ", "ij,jk->ik", [{3, 4}, {4, 5}]}
  ]

  @lanes [:eager, :evaluator, :native, :fallback]

  for {{eq, explicit, shapes}, i} <- Enum.with_index(@cases), lane <- @lanes do
    test "#{inspect(eq)} on #{inspect(shapes)} (#{lane}, case #{i})" do
      eq = unquote(eq)
      operands = operands(unquote(Macro.escape(shapes)), :f32)
      expected = reference(unquote(explicit), operands)

      got = run(unquote(lane), eq, operands)

      assert Nx.type(got) == {:f, 32}
      assert_close(got, expected, tol: @f32_tol)
    end
  end

  describe "operand counts beyond the block arity table" do
    for lane <- @lanes do
      test "17 operands (#{lane})" do
        operands =
          for i <- 1..17,
              do: Nx.iota({3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(i) |> Nx.divide(8)

        eq = Enum.map_join(1..17, ",", fn _ -> "i" end) <> "->i"
        expected = reference(eq, operands)

        assert_close(run(unquote(lane), eq, operands), expected, tol: @f32_tol)
      end
    end
  end

  describe "dtypes" do
    for lane <- @lanes do
      test "integer operands without a contraction stay exact (#{lane})" do
        for {eq, shapes} <- [
              {"ij,ij->ij", [{3, 4}, {3, 4}]},
              {"ii->i", [{4, 4}]},
              {"ij->j", [{3, 4}]}
            ] do
          operands = operands(shapes, :s32)
          got = run(unquote(lane), eq, operands)

          assert Nx.type(got) == {:s, 32}
          assert Nx.to_flat_list(to_binary(got)) == Nx.to_flat_list(reference(eq, operands))
        end
      end

      test "result keeps the operands' narrow integer type (#{lane})" do
        t = Nx.tensor([[1, 2], [3, 4]], type: :u8, backend: Nx.BinaryBackend)
        got = run(unquote(lane), "ij->j", [t])

        assert Nx.type(got) == {:u, 8}
        assert Nx.to_flat_list(to_binary(got)) == [4, 6]
      end

      test "mixed float types promote like Nx.dot (#{lane})" do
        [a, b] = operands([{3, 4}, {4, 5}], :f32)
        operands = [Nx.as_type(a, :bf16), b]
        got = run(unquote(lane), "ij,jk->ik", operands)

        assert Nx.type(got) == {:f, 32}
        assert_close(got, reference("ij,jk->ik", operands), tol: @f32_tol)
      end
    end
  end

  describe "integer contractions" do
    test "run exactly on the composed path" do
      operands = operands([{3, 4}, {4, 5}], :s32)
      got = run(:fallback, "ij,jk->ik", operands)

      assert Nx.type(got) == {:s, 32}
      assert Nx.to_flat_list(got) == Nx.to_flat_list(reference("ij,jk->ik", operands))
    end

    # MLX's matmul is float-only, so Emily rejects integer contractions
    # the same way it rejects an integer Nx.dot.
    for lane <- [:eager, :evaluator, :native] do
      test "raise MLX's float-only matmul error on Emily (#{lane})" do
        operands = operands([{3, 4}, {4, 5}], :s32)

        assert_raise ArgumentError, ~r/Only inexact types are supported/, fn ->
          run(unquote(lane), "ij,jk->ik", operands)
        end
      end
    end
  end

  describe "defn integration" do
    test "callable from a caller's own defn" do
      [q, k] = operands([{1, 2, 3, 4}, {1, 2, 5, 4}], :f32)
      expected = Nx.multiply(reference("bhid,bhjd->bhij", [q, k]), 0.5)

      for opts <- [[native: false], [native: true, native_fallback: :raise]] do
        got =
          Nx.Defn.jit_apply(
            &Defns.attention_scores/2,
            [to_emily(q), to_emily(k)],
            [compiler: Emily.Compiler] ++ opts
          )

        assert_close(got, expected, tol: @f32_tol)
      end

      assert_close(FastFallback.run(&Defns.attention_scores/2, [q, k]), expected, tol: @f32_tol)
    end

    test "eager call accepts operands on a mix of backends" do
      [a, b] = operands([{3, 4}, {4, 5}], :f32)
      got = Emily.Fast.einsum("ij,jk->ik", [to_emily(a), b])

      assert %Nx.Tensor{data: %Emily.Backend{}} = got
      assert_close(got, reference("ij,jk->ik", [a, b]), tol: @f32_tol)
    end
  end

  describe "invalid equations" do
    @invalid [
      {"ij,jk->ik", [{2, 3}], ~r/2 input subscript\(s\) but 1 operand/},
      {"i1->i", [{2, 3}], ~r/contains "1"/},
      {"...i...->i", [{2, 3}], ~r/more than one ellipsis/},
      {"i.j->i", [{2, 3}], ~r/contains "."/},
      {"ij->ii", [{2, 3}], ~r/repeats a label in the output/},
      {"ij->k", [{2, 3}], ~r/"k".*do not appear/},
      {"ij->i->j", [{2, 3}], ~r/more than one "->"/},
      {"ijk->i", [{2, 3}], ~r/rank 2/},
      {"...ijk->i", [{2, 3}], ~r/at least 3 dims/},
      {"ii->i", [{2, 3}], ~r/repeats label "i".*different sizes/},
      {"ij,jk->ik", [{2, 3}, {4, 5}], ~r/cannot broadcast label "j"/}
    ]

    for {{eq, shapes, message}, i} <- Enum.with_index(@invalid), lane <- @lanes do
      test "#{inspect(eq)} on #{inspect(shapes)} raises (#{lane}, case #{i})" do
        operands = operands(unquote(Macro.escape(shapes)), :f32)

        assert_raise ArgumentError, unquote(Macro.escape(message)), fn ->
          run(unquote(lane), unquote(eq), operands)
        end
      end
    end

    test "requires at least one operand" do
      assert_raise ArgumentError, ~r/at least one operand/, fn ->
        Emily.Fast.einsum("->", [])
      end
    end

    test "requires a string equation and a list of operands" do
      assert_raise ArgumentError, ~r/expects an equation string/, fn ->
        Emily.Fast.einsum(~c"ij->ji", [Nx.iota({2, 2})])
      end
    end
  end

  # ---- lanes ----

  defp run(:eager, eq, operands), do: Emily.Fast.einsum(eq, Enum.map(operands, &to_emily/1))

  defp run(:evaluator, eq, operands),
    do: jit_emily(eq, operands, native: false)

  defp run(:native, eq, operands),
    do: jit_emily(eq, operands, native: true, native_fallback: :raise)

  defp run(:fallback, eq, operands),
    do:
      FastFallback.run(fn ops -> Emily.Fast.einsum(eq, Tuple.to_list(ops)) end, [
        List.to_tuple(operands)
      ])

  defp jit_emily(eq, operands, opts) do
    Nx.Defn.jit_apply(
      fn ops -> Emily.Fast.einsum(eq, Tuple.to_list(ops)) end,
      [operands |> Enum.map(&to_emily/1) |> List.to_tuple()],
      [compiler: Emily.Compiler] ++ opts
    )
  end

  # ---- helpers ----

  # Small, mixed-sign, non-symmetric values so a wrong axis pairing or a
  # dropped diagonal changes the result.
  defp operands(shapes, type) do
    shapes
    |> Enum.with_index(1)
    |> Enum.map(fn {shape, i} ->
      t =
        Nx.iota(shape, backend: Nx.BinaryBackend)
        |> Nx.multiply(i + 2)
        |> Nx.remainder(7)
        |> Nx.subtract(3)

      if Nx.Type.integer?(Nx.Type.normalize!(type)),
        do: Nx.as_type(t, type),
        else: t |> Nx.as_type(type) |> Nx.divide(4)
    end)
  end

  defp to_emily(t), do: Nx.backend_copy(t, Emily.Backend)
  defp to_binary(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  # Brute-force einsum over an explicit, ellipsis-free equation: visit
  # every assignment of values to labels and accumulate the product of
  # the matching operand entries into the output entry. A size-1 operand
  # dim reads index 0 (broadcasting).
  defp reference(explicit, operands) do
    [lhs, rhs] = String.split(explicit, "->")
    terms = lhs |> String.split(",") |> Enum.map(&String.graphemes/1)
    out = String.graphemes(rhs)
    operands = Enum.map(operands, &to_binary/1)

    sizes =
      terms
      |> Enum.zip(operands)
      |> Enum.flat_map(fn {labels, t} -> Enum.zip(labels, Tuple.to_list(Nx.shape(t))) end)
      |> Enum.reduce(%{}, fn {l, d}, acc -> Map.update(acc, l, d, &max(&1, d)) end)

    labels = Map.keys(sizes)
    type = operands |> Enum.map(&Nx.type/1) |> Enum.reduce(&Nx.Type.merge/2)

    flat =
      Enum.map(operands, fn t ->
        {t |> Nx.to_flat_list() |> List.to_tuple(), Tuple.to_list(Nx.shape(t))}
      end)

    acc =
      labels
      |> assignments(sizes)
      |> Enum.reduce(%{}, fn assign, acc ->
        product =
          terms
          |> Enum.zip(flat)
          |> Enum.reduce(1, fn {term, {data, dims}}, prod ->
            prod * elem(data, flat_index(term, dims, assign))
          end)

        Map.update(acc, Enum.map(out, &assign[&1]), product, &(&1 + product))
      end)

    out_shape = out |> Enum.map(&sizes[&1]) |> List.to_tuple()

    out
    |> assignments(sizes)
    |> Enum.map(fn assign -> Map.fetch!(acc, Enum.map(out, &assign[&1])) end)
    |> Nx.tensor(type: type, backend: Nx.BinaryBackend)
    |> Nx.reshape(out_shape)
  end

  # Every map from `keys` to an index below its size, in row-major order.
  defp assignments(keys, sizes) do
    Enum.reduce(Enum.reverse(keys), [%{}], fn key, acc ->
      for v <- 0..(sizes[key] - 1)//1, rest <- acc, do: Map.put(rest, key, v)
    end)
  end

  defp flat_index(term, dims, assign) do
    term
    |> Enum.zip(dims)
    |> Enum.reduce(0, fn {label, dim}, idx ->
      i = if dim == 1, do: 0, else: assign[label]
      idx * dim + i
    end)
  end
end
