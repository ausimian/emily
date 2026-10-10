defmodule Emily.Fast.Einsum do
  @moduledoc false
  # Einsum-equation parser and composed-Nx fallback behind
  # `Emily.Fast.einsum/2`.
  #
  # `parse!/2` mirrors `mx::einsum`'s own parser (deps/mlx_src/mlx/einsum.cpp)
  # so the fallback accepts exactly the equations the MLX kernel does and
  # gives them the same meaning:
  #
  #   * spaces are ignored;
  #   * without `->`, the output is every label that appears exactly once,
  #     sorted by codepoint (so uppercase before lowercase), preceded by the
  #     ellipsis dims if any input has an ellipsis;
  #   * `...` expands right-aligned, using the unused letters from the end of
  #     `a..zA..Z`, so a shorter ellipsis broadcasts against a longer one;
  #   * a size-1 dim broadcasts against any other size for the same label;
  #   * a label repeated within one input takes that input's diagonal, and
  #     its sizes must match exactly;
  #   * an input label missing from the output is summed out.
  #
  # The parsed spec is fully explicit and ellipsis-free. `equation/1` renders
  # it back to the string the Emily kernel receives, so both paths run the
  # same normalised equation.
  #
  # `compose/3` contracts the operands pairwise, left to right, with
  # `Nx.dot/6`: labels still needed by the output or by a later operand
  # stay as batch axes, other shared labels are contracted, and labels no
  # longer needed are summed out as early as possible. It does not search
  # for a cheaper contraction order the way MLX's path optimiser does, so
  # results match up to floating-point reassociation.

  defstruct [:inputs, :output, :sizes]

  @letters Enum.to_list(?a..?z) ++ Enum.to_list(?A..?Z)

  @doc """
  Parses `subscripts` against the operand `shapes`. Raises `ArgumentError`
  on any equation `mx::einsum` would reject.
  """
  def parse!(subscripts, shapes) when is_binary(subscripts) and is_list(shapes) do
    if shapes == [] do
      raise ArgumentError, "Emily.Fast.einsum/2: at least one operand is required"
    end

    eq = String.replace(subscripts, " ", "")
    {terms, out_term} = split_terms!(eq, length(shapes), subscripts)
    spare = Enum.reject(@letters, &(&1 in String.to_charlist(eq)))

    {inputs, ellipsis_len} =
      terms
      |> Enum.zip(shapes)
      |> Enum.with_index()
      |> Enum.map_reduce(0, fn {{term, shape}, i}, max_len ->
        len = ellipsis_length!(term, shape, i, subscripts)
        {expand(term, len, spare, subscripts), max(len, max_len)}
      end)

    output = expand(out_term, ellipsis_len, spare, subscripts)
    sizes = label_sizes!(inputs, shapes, subscripts)
    check_output!(output, sizes, subscripts)

    %__MODULE__{inputs: inputs, output: output, sizes: sizes}
  end

  # Tokenized input terms plus the output term, inferred when implicit.
  defp split_terms!(eq, n_operands, subscripts) do
    {lhs, rhs} =
      case String.split(eq, "->") do
        [lhs] -> {lhs, nil}
        [lhs, rhs] -> {lhs, rhs}
        _ -> fail!(subscripts, "found more than one \"->\"")
      end

    terms = lhs |> String.split(",") |> Enum.map(&tokenize!(&1, subscripts))

    if length(terms) != n_operands do
      fail!(
        subscripts,
        "has #{length(terms)} input subscript(s) but #{n_operands} operand(s) were given"
      )
    end

    {terms, if(rhs, do: tokenize!(rhs, subscripts), else: implicit_output(terms))}
  end

  defp check_output!(output, sizes, subscripts) do
    if Enum.uniq(output) != output do
      fail!(subscripts, "repeats a label in the output")
    end

    case Enum.reject(output, &Map.has_key?(sizes, &1)) do
      [] ->
        :ok

      missing ->
        fail!(
          subscripts,
          "output label(s) #{inspect(List.to_string(missing))} " <>
            "do not appear in any input"
        )
    end
  end

  @doc "The explicit, ellipsis-free equation for a parsed spec."
  def equation(%__MODULE__{inputs: inputs, output: output}) do
    Enum.map_join(inputs, ",", &List.to_string/1) <> "->" <> List.to_string(output)
  end

  @doc "The output shape of a parsed spec."
  def output_shape(%__MODULE__{output: output, sizes: sizes}) do
    output |> Enum.map(&Map.fetch!(sizes, &1)) |> List.to_tuple()
  end

  # A term is a list of label codepoints with at most one `:ellipsis`.
  defp tokenize!(term, subscripts), do: tokenize(term, [], false, subscripts)

  defp tokenize("", acc, _seen?, _subscripts), do: Enum.reverse(acc)

  defp tokenize("..." <> _rest, _acc, true, subscripts),
    do: fail!(subscripts, "has more than one ellipsis in a single subscript")

  defp tokenize("..." <> rest, acc, false, subscripts),
    do: tokenize(rest, [:ellipsis | acc], true, subscripts)

  defp tokenize(<<c, rest::binary>>, acc, seen?, subscripts) when c in @letters,
    do: tokenize(rest, [c | acc], seen?, subscripts)

  defp tokenize(rest, _acc, _seen?, subscripts),
    do:
      fail!(
        subscripts,
        "contains #{inspect(String.first(rest))}; subscripts must be letters " <>
          "a-z, A-Z or a single \"...\""
      )

  defp implicit_output(terms) do
    labels = terms |> List.flatten() |> Enum.reject(&(&1 == :ellipsis))

    once =
      labels |> Enum.frequencies() |> Enum.filter(&(elem(&1, 1) == 1)) |> Enum.map(&elem(&1, 0))

    prefix = if Enum.any?(terms, &(:ellipsis in &1)), do: [:ellipsis], else: []
    prefix ++ Enum.sort(once)
  end

  defp ellipsis_length!(term, shape, i, subscripts) do
    rank = tuple_size(shape)
    named = Enum.count(term, &(&1 != :ellipsis))

    cond do
      :ellipsis not in term and named != rank ->
        fail!(
          subscripts,
          "gives operand #{i} #{named} label(s), but its shape " <>
            "#{inspect(shape)} has rank #{rank}"
        )

      :ellipsis in term and named > rank ->
        fail!(
          subscripts,
          "needs at least #{named} dims for operand #{i}, but its " <>
            "shape #{inspect(shape)} has rank #{rank}"
        )

      :ellipsis in term ->
        rank - named

      true ->
        0
    end
  end

  defp expand(term, len, spare, subscripts) do
    if len > length(spare) do
      fail!(
        subscripts,
        "needs #{len} ellipsis label(s) but only #{length(spare)} " <>
          "unused letters remain"
      )
    end

    Enum.flat_map(term, fn
      :ellipsis -> Enum.take(spare, -len)
      c -> [c]
    end)
  end

  defp label_sizes!(inputs, shapes, subscripts) do
    inputs
    |> Enum.zip(shapes)
    |> Enum.with_index()
    |> Enum.reduce(%{}, fn {{labels, shape}, i}, sizes ->
      dims = Tuple.to_list(shape)
      check_repeats!(labels, dims, i, subscripts)

      Enum.zip(labels, dims)
      |> Enum.reduce(sizes, &merge_size!(&1, &2, i, subscripts))
    end)
  end

  # A size-1 dim broadcasts against any size; other sizes must agree.
  defp merge_size!({label, dim}, sizes, i, subscripts) do
    case Map.fetch(sizes, label) do
      :error ->
        Map.put(sizes, label, dim)

      {:ok, seen} when seen == dim or seen == 1 or dim == 1 ->
        Map.put(sizes, label, max(seen, dim))

      {:ok, seen} ->
        fail!(
          subscripts,
          "cannot broadcast label #{inspect(<<label>>)} of " <>
            "operand #{i} (size #{dim}) against size #{seen}"
        )
    end
  end

  defp check_repeats!(labels, dims, i, subscripts) do
    Enum.zip(labels, dims)
    |> Enum.group_by(&elem(&1, 0), &elem(&1, 1))
    |> Enum.each(fn {label, label_dims} ->
      if length(Enum.uniq(label_dims)) > 1 do
        fail!(
          subscripts,
          "repeats label #{inspect(<<label>>)} in operand #{i} " <>
            "over dims of different sizes #{inspect(label_dims)}"
        )
      end
    end)
  end

  defp fail!(subscripts, message) do
    raise ArgumentError, "Emily.Fast.einsum/2: equation #{inspect(subscripts)} #{message}"
  end

  # =================================================================
  # Composed fallback
  # =================================================================

  @doc """
  Evaluates a parsed spec with composed Nx ops and casts the result to
  `type`. Works on any backend and inside `defn`.
  """
  def compose(%__MODULE__{} = spec, tensors, type) do
    [{t, labels} | rest] =
      Enum.zip_with(tensors, spec.inputs, fn t, labels ->
        {t, labels} = take_diagonals(t, labels)
        {broadcast_labels(t, labels, spec.sizes), labels}
      end)

    {t, labels} = contract(t, labels, rest, spec.output)
    {t, labels} = sum_out(t, labels, spec.output)
    perm = Enum.map(spec.output, fn l -> Enum.find_index(labels, &(&1 == l)) end)

    t
    |> transpose_if_needed(perm)
    |> Nx.as_type(type)
  end

  # Replace each repeated label's pair of axes with their diagonal: move
  # the pair to the end, flatten it to n*n, and keep every (n+1)-th entry.
  defp take_diagonals(t, labels) do
    case first_repeat(labels) do
      nil ->
        {t, labels}

      {label, p, q} ->
        others = Enum.to_list(0..(length(labels) - 1)//1) -- [p, q]
        n = Nx.axis_size(t, p)
        lead = Enum.map(others, &Nx.axis_size(t, &1))

        t =
          t
          |> Nx.transpose(axes: others ++ [p, q])
          |> Nx.reshape(List.to_tuple(lead ++ [n * n]))
          |> Nx.slice_along_axis(0, n * n, axis: -1, strides: n + 1)

        take_diagonals(t, Enum.map(others, &Enum.at(labels, &1)) ++ [label])
    end
  end

  defp first_repeat(labels) do
    labels
    |> Enum.with_index()
    |> Enum.find_value(fn {label, p} ->
      q = labels |> Enum.drop(p + 1) |> Enum.find_index(&(&1 == label))
      if q, do: {label, p, p + 1 + q}
    end)
  end

  defp broadcast_labels(t, labels, sizes) do
    target = labels |> Enum.map(&Map.fetch!(sizes, &1)) |> List.to_tuple()
    if Nx.shape(t) == target, do: t, else: Nx.broadcast(t, target)
  end

  defp contract(t, labels, [], _output), do: {t, labels}

  defp contract(a, la, [{b, lb} | rest], output) do
    keep = output ++ Enum.flat_map(rest, &elem(&1, 1))
    {a, la} = sum_out(a, la, keep ++ lb)
    {b, lb} = sum_out(b, lb, keep ++ la)

    shared = Enum.filter(la, &(&1 in lb))
    batch = Enum.filter(shared, &(&1 in keep))

    {a, la} = batch_first(a, la, batch)
    {b, lb} = batch_first(b, lb, batch)

    contract_a = for l <- shared, l not in batch, do: Enum.find_index(la, &(&1 == l))
    contract_b = for l <- shared, l not in batch, do: Enum.find_index(lb, &(&1 == l))
    batch_axes = Enum.to_list(0..(length(batch) - 1)//1)

    t = Nx.dot(a, contract_a, batch_axes, b, contract_b, batch_axes)
    contract(t, batch ++ (la -- shared) ++ (lb -- shared), rest, output)
  end

  # Move the batch labels to the front, in `batch` order (Nx.dot/6 wants
  # batch axes leading and aligned across both operands).
  defp batch_first(t, labels, batch) do
    order = batch ++ (labels -- batch)
    perm = Enum.map(order, fn l -> Enum.find_index(labels, &(&1 == l)) end)
    {transpose_if_needed(t, perm), order}
  end

  # Sum out every axis whose label is not in `keep`.
  defp sum_out(t, labels, keep) do
    case for {l, i} <- Enum.with_index(labels), l not in keep, do: i do
      [] -> {t, labels}
      axes -> {Nx.sum(t, axes: axes), Enum.filter(labels, &(&1 in keep))}
    end
  end

  defp transpose_if_needed(t, perm) do
    if perm == Enum.to_list(0..(length(perm) - 1)//1), do: t, else: Nx.transpose(t, axes: perm)
  end
end
