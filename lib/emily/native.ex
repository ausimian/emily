defmodule Emily.Native do
  @moduledoc false
  # Thin NIF loader for the emily C++ shim. Every function here maps
  # directly to one NIF in c_src/emily_nif.cpp. No policy, no caching,
  # no defaults — higher layers do that.

  @on_load :__on_load__
  @compile {:autoload, false}

  @doc false
  def __on_load__ do
    path = :filename.join(:code.priv_dir(:emily), ~c"libemily")
    :erlang.load_nif(path, 0)
  end

  # --- M0 surface --------------------------------------------------

  @doc """
  Build a lazy MLX tensor from a raw binary, shape, and dtype.
  """
  @spec from_binary(binary(), [non_neg_integer()], {atom(), non_neg_integer()}) ::
          reference()
  def from_binary(_data, _shape, _dtype), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Force evaluation of the lazy graph rooted at `tensor`, then return
  the materialized bytes as a binary.
  """
  @spec to_binary(reference()) :: binary()
  def to_binary(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Return the tensor's shape as a list of ints."
  @spec shape(reference()) :: [non_neg_integer()]
  def shape(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Return the tensor's dtype as an `{atom, bits}` tuple."
  @spec dtype(reference()) :: {atom(), non_neg_integer()}
  def dtype(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Force evaluation of the lazy graph rooted at `tensor`."
  @spec eval(reference()) :: :ok
  def eval(_tensor), do: :erlang.nif_error(:nif_not_loaded)
end
