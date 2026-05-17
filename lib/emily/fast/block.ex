defmodule Emily.Fast.Block do
  @moduledoc """
  Block-dispatch structs for `Emily.Fast.*` fused kernels.

  Each `Emily.Fast.*` helper emits an `Nx.block/4` node carrying one
  of these structs. `Emily.Backend.block/4` pattern-matches on the
  struct to call the matching `mx::fast::*` NIF; non-Emily backends
  fall through to the default `fun` supplied by `Nx.block/4`, which
  runs the composed-defn fallback baked into each helper.

  Static configuration (eps, dims, scale, causal, …) is carried as
  struct fields. Runtime tensors travel in the `Nx.block/4` args list.
  """

  defmodule RMSNorm do
    @moduledoc false
    defstruct eps: 1.0e-6
  end

  defmodule LayerNorm do
    @moduledoc false
    defstruct eps: 1.0e-5
  end

  defmodule RoPE do
    @moduledoc false
    defstruct dims: nil, traditional: false, base: 10_000.0, scale: 1.0
  end

  defmodule RoPEWithFreqs do
    @moduledoc false
    defstruct dims: nil, traditional: false, scale: 1.0
  end

  defmodule SDPA do
    @moduledoc false
    defstruct scale: nil, causal: false
  end

  defmodule SDPAWithSinks do
    @moduledoc false
    defstruct scale: nil, causal: false
  end

  defmodule SDPAWithMask do
    @moduledoc false
    defstruct scale: nil
  end

  defmodule SDPAWithMaskAndSinks do
    @moduledoc false
    defstruct scale: nil
  end
end
