defmodule Emily.BackendGenerators do
  @moduledoc """
  StreamData generators for property-testing `Emily.Backend` against
  `Nx.BinaryBackend`. Kept intentionally small: shape rank ≤ 4, dim ≤ 6,
  values bounded so that f32 arithmetic doesn't overflow/underflow on
  reductions.
  """

  import StreamData

  @max_rank 4
  @max_dim 6

  @doc "Generate a shape tuple (rank ≤ 4, each dim 1..6)."
  def shape do
    bind(integer(0..@max_rank), fn rank ->
      list_of(integer(1..@max_dim), length: rank) |> map(&List.to_tuple/1)
    end)
  end

  @doc "Generate a non-scalar shape (rank ≥ 1)."
  def non_scalar_shape do
    bind(integer(1..@max_rank), fn rank ->
      list_of(integer(1..@max_dim), length: rank) |> map(&List.to_tuple/1)
    end)
  end

  @doc "Generate a shape with exactly the given rank."
  def shape(rank) when is_integer(rank) do
    list_of(integer(1..@max_dim), length: rank) |> map(&List.to_tuple/1)
  end

  @doc """
  Generate a random tensor on `Nx.BinaryBackend`. Values are bounded so
  that typical reductions don't overflow.
  """
  def tensor(shape, {:f, 32}) do
    list_of(float(min: -10.0, max: 10.0), length: Nx.size(shape))
    |> map(&Nx.tensor(&1, type: {:f, 32}, backend: Nx.BinaryBackend))
    |> map(&Nx.reshape(&1, shape))
  end

  def tensor(shape, {:bf, 16}) do
    list_of(float(min: -10.0, max: 10.0), length: Nx.size(shape))
    |> map(&Nx.tensor(&1, type: {:f, 32}, backend: Nx.BinaryBackend))
    |> map(&Nx.reshape(&1, shape))
    |> map(&Nx.as_type(&1, {:bf, 16}))
  end

  def tensor(shape, {:f, 16}) do
    list_of(float(min: -10.0, max: 10.0), length: Nx.size(shape))
    |> map(&Nx.tensor(&1, type: {:f, 32}, backend: Nx.BinaryBackend))
    |> map(&Nx.reshape(&1, shape))
    |> map(&Nx.as_type(&1, {:f, 16}))
  end

  def tensor(shape, {:s, bits}) do
    max = min(100, Bitwise.bsl(1, bits - 1) - 1)
    min = -max

    list_of(integer(min..max), length: Nx.size(shape))
    |> map(&Nx.tensor(&1, type: {:s, bits}, backend: Nx.BinaryBackend))
    |> map(&Nx.reshape(&1, shape))
  end

  def tensor(shape, {:u, bits}) do
    max = min(100, Bitwise.bsl(1, bits) - 1)

    list_of(integer(0..max), length: Nx.size(shape))
    |> map(&Nx.tensor(&1, type: {:u, bits}, backend: Nx.BinaryBackend))
    |> map(&Nx.reshape(&1, shape))
  end

  @doc "Transfer a BinaryBackend tensor to Emily.Backend."
  def to_emily(tensor), do: Nx.backend_transfer(tensor, Emily.Backend)

  @doc "Transfer an Emily.Backend tensor to BinaryBackend."
  def to_bin(tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  @doc "Tolerance by dtype, for assert_close."
  def tol_for({:f, _}), do: 1.0e-4
  def tol_for({:bf, _}), do: 1.0e-2
  def tol_for(_), do: 0.0

  @doc """
  Assert two tensors (Nx or Emily) match element-wise within tolerance.

  Both are materialised to Elixir lists of numbers via Nx.to_flat_list.
  """
  def assert_close(actual, expected, opts \\ [])

  def assert_close(%Nx.Tensor{} = actual, %Nx.Tensor{} = expected, opts) do
    tol = opts[:tol] || tol_for(expected.type)

    assert_shape(actual, expected)

    actual_list = actual |> to_bin() |> Nx.to_flat_list()
    expected_list = expected |> to_bin() |> Nx.to_flat_list()

    assert_close_lists(actual_list, expected_list, tol)
  end

  defp assert_shape(%Nx.Tensor{shape: s}, %Nx.Tensor{shape: s}), do: :ok

  defp assert_shape(%Nx.Tensor{shape: s1}, %Nx.Tensor{shape: s2}) do
    ExUnit.Assertions.flunk("shape mismatch: #{inspect(s1)} vs #{inspect(s2)}")
  end

  defp assert_close_lists(actual, expected, tol) do
    pairs = Enum.zip(actual, expected)

    mismatches =
      pairs
      |> Enum.with_index()
      |> Enum.filter(fn {{a, e}, _} -> not close?(a, e, tol) end)
      |> Enum.take(5)

    if mismatches == [] do
      :ok
    else
      details =
        Enum.map_join(mismatches, "\n  ", fn {{a, e}, i} ->
          "[#{i}] actual=#{inspect(a)} expected=#{inspect(e)} diff=#{inspect(abs_diff(a, e))}"
        end)

      ExUnit.Assertions.flunk("""
      tensor mismatch (tol=#{tol}); first mismatches:
        #{details}
      """)
    end
  end

  defp close?(a, b, tol) when is_number(a) and is_number(b) and tol == 0 do
    a === b
  end

  defp close?(a, b, tol) when is_number(a) and is_number(b) do
    abs(a - b) <= tol + tol * abs(b)
  end

  defp close?(a, b, _tol), do: a == b

  defp abs_diff(a, b) when is_number(a) and is_number(b), do: abs(a - b)
  defp abs_diff(_, _), do: :na

  # --- Linalg generators ---

  @doc "Generate a random n×n square matrix (n in 2..5)."
  def square_matrix do
    bind(integer(2..5), fn n ->
      tensor({n, n}, {:f, 32})
    end)
  end

  @doc """
  Generate a symmetric positive-definite matrix by constructing A^T A + nI.
  The identity diagonal ensures positive eigenvalues even if A is rank-deficient.
  """
  def spd_matrix do
    bind(integer(2..5), fn n ->
      tensor({n, n}, {:f, 32})
      |> map(fn a ->
        ata = Nx.dot(Nx.transpose(a), a)
        Nx.add(ata, Nx.multiply(Nx.eye(n, backend: Nx.BinaryBackend), n))
      end)
    end)
  end

  @doc "Generate a symmetric matrix (A + A^T) / 2."
  def symmetric_matrix do
    bind(integer(2..5), fn n ->
      tensor({n, n}, {:f, 32})
      |> map(fn a -> Nx.divide(Nx.add(a, Nx.transpose(a)), 2) end)
    end)
  end

  @doc "Generate a lower-triangular matrix with positive diagonal (non-singular)."
  def lower_triangular_matrix do
    bind(integer(2..5), fn n ->
      tensor({n, n}, {:f, 32})
      |> map(fn a ->
        # Strictly-lower triangle + positive diagonal.
        strict_lower = Nx.tril(a, k: -1)
        diag = a |> Nx.take_diagonal() |> Nx.abs() |> Nx.add(1.0) |> Nx.make_diagonal()
        Nx.add(strict_lower, diag)
      end)
    end)
  end

  @doc "Add diagonal dominance to a square matrix to ensure non-singularity."
  def make_well_conditioned(a) do
    n = elem(Nx.shape(a), 0)
    # Elements are in [-10, 10]. Row sum of abs(off-diagonal) ≤ (n-1)*10.
    # Diagonal must strictly exceed that, so we add (n-1)*10 + 10 = n*10
    # per element — but that lands right on the boundary for n=2 (diag=10,
    # off-diag sum=10). Use n*10 + 20 for clear margin under f32 rounding.
    Nx.add(a, Nx.multiply(Nx.eye(n, backend: Nx.BinaryBackend), n * 10 + 20))
  end
end
