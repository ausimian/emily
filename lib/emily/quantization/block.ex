defmodule Emily.Quantization.Block do
  @moduledoc false
  # Block-dispatch struct for the fused quantized matmul kernel, mirroring
  # the `Emily.Fast.Block.*` pattern. Static config (transpose / group_size
  # / bits / mode) lives on the struct; the runtime tensors — the
  # activation `x` plus the `QuantizedWeight`'s packed value, scales, and
  # biases — travel in the `Nx.block/4` args list.
  #
  # `Emily.Quantization.quantized_matmul_defn/2` emits the block;
  # `Emily.Backend.block/4` dispatches it to `Native.quantized_matmul`,
  # and the Expr compiler lowers it to the `quantized_matmul` opcode.
  # Non-Emily backends run the composed `Nx.dot(x, dequantize_defn(qw))`
  # fallback baked into the block.

  defmodule QuantizedMatmul do
    @moduledoc false
    defstruct [:transpose, :group_size, :bits, :mode]
  end
end
