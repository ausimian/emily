defmodule Emily.FastKernelsRewrites do
  @moduledoc """
  Counts the layers `Emily.Bumblebee.FastKernels.apply/2` rewrote in an
  Axon model, by the name of the fused impl each rewritten node calls
  (`:fast_layer_norm_impl`, `:fast_rms_norm_impl`, `:fast_rope_impl`,
  `:fast_sdpa_impl`).

  The `:fast_kernels_full` tests assert these counts so that a model the
  shim silently stops rewriting fails loudly, instead of passing because
  the unrewritten model still matches the reference.
  """

  alias Emily.Bumblebee.FastKernels

  @spec counts(Axon.t()) :: %{atom() => pos_integer()}
  def counts(%Axon{nodes: nodes}) do
    nodes
    |> Map.values()
    |> Enum.flat_map(fn %Axon.Node{op: op} ->
      with true <- is_function(op),
           info = Function.info(op),
           FastKernels <- info[:module] do
        [info[:name]]
      else
        _ -> []
      end
    end)
    |> Enum.frequencies()
  end
end
