defmodule Emily.BackendConvTest do
  @moduledoc """
  Oracle tests for `Emily.Backend.conv/4`. Each case runs the same
  `Nx.conv` on `Emily.Backend` and `Nx.BinaryBackend` and asserts the
  outputs agree within a dtype-appropriate tolerance.

  Covers the layout mapping (Nx NCHW/OIHW ↔ MLX NHWC/OHWI), the
  non-default permutation paths, grouping (including depthwise),
  dilation, padding variants, and integer-operand `Nx.astype` coercion.
  """

  use ExUnit.Case, async: true

  import Emily.BackendGenerators, only: [assert_close: 2]

  defp emily(tensor), do: Nx.backend_transfer(tensor, Emily.Backend)
  defp bin(tensor), do: Nx.backend_transfer(tensor, Nx.BinaryBackend)

  # Build a pair of (input, kernel) on BinaryBackend with deterministic
  # values scaled so accumulation stays in the f32 sweet spot.
  defp inputs(input_shape, kernel_shape) do
    scale = 0.1

    input =
      input_shape
      |> Nx.iota(type: {:f, 32}, backend: Nx.BinaryBackend)
      |> Nx.multiply(scale)

    kernel =
      kernel_shape
      |> Nx.iota(type: {:f, 32}, backend: Nx.BinaryBackend)
      |> Nx.multiply(scale)

    {input, kernel}
  end

  defp run(input, kernel, opts) do
    emily_result = Nx.conv(emily(input), emily(kernel), opts)
    ref_result = Nx.conv(bin(input), bin(kernel), opts)
    assert_close(emily_result, ref_result)
  end

  describe "rank & default layout" do
    test "1-D conv, stride 1, padding :valid" do
      # {batch, channels, length} × {out, in, width}
      {input, kernel} = inputs({2, 3, 8}, {4, 3, 3})
      run(input, kernel, [])
    end

    test "2-D conv, default NCHW, stride 1, padding :valid" do
      {input, kernel} = inputs({2, 3, 5, 5}, {4, 3, 3, 3})
      run(input, kernel, [])
    end

    test "3-D conv (rank-5) smoke" do
      {input, kernel} = inputs({1, 2, 4, 4, 4}, {3, 2, 2, 2, 2})
      run(input, kernel, [])
    end
  end

  describe "strides, padding, dilation" do
    test "2-D conv with stride 2" do
      {input, kernel} = inputs({1, 2, 6, 6}, {4, 2, 3, 3})
      run(input, kernel, strides: 2)
    end

    test "2-D conv with padding :same" do
      {input, kernel} = inputs({1, 2, 5, 5}, {3, 2, 3, 3})
      run(input, kernel, padding: :same)
    end

    test "2-D conv with explicit asymmetric padding" do
      {input, kernel} = inputs({1, 2, 5, 5}, {3, 2, 3, 3})
      run(input, kernel, padding: [{1, 2}, {0, 1}])
    end

    test "2-D conv with kernel_dilation > 1" do
      {input, kernel} = inputs({1, 2, 7, 7}, {3, 2, 3, 3})
      run(input, kernel, kernel_dilation: 2)
    end

    test "2-D conv with input_dilation > 1 (transposed conv)" do
      {input, kernel} = inputs({1, 2, 4, 4}, {3, 2, 3, 3})
      run(input, kernel, input_dilation: 2)
    end
  end

  describe "grouping" do
    test "feature_group_size: 2" do
      # in_channels = 4 split into 2 groups; each group contributes to
      # 3 of the 6 output filters.
      {input, kernel} = inputs({1, 4, 5, 5}, {6, 2, 3, 3})
      run(input, kernel, feature_group_size: 2)
    end

    test "depthwise conv (feature_group_size == in_channels)" do
      # in_channels = 4 → 4 groups (each channel independent);
      # kernel in-channels-per-group = 1; out_channels = 4.
      {input, kernel} = inputs({1, 4, 5, 5}, {4, 1, 3, 3})
      run(input, kernel, feature_group_size: 4)
    end
  end

  describe "non-default permutations" do
    test "input_permutation: [0, 3, 1, 2] (caller passes NHWC input)" do
      # Start from canonical NCHW, transpose to NHWC, then tell Nx about it.
      {nchw_input, kernel} = inputs({1, 3, 5, 5}, {4, 3, 3, 3})
      nhwc_input = Nx.transpose(nchw_input, axes: [0, 2, 3, 1])

      run(nhwc_input, kernel, input_permutation: [0, 3, 1, 2])
    end

    test "kernel_permutation: [3, 2, 0, 1] (caller passes HWIO kernel)" do
      {input, oihw_kernel} = inputs({1, 3, 5, 5}, {4, 3, 3, 3})
      # HWIO: [height, width, in, out]; pull axes so that original [0,1,2,3]
      # (O, I, H, W) becomes [2, 3, 1, 0] (H, W, I, O).
      hwio_kernel = Nx.transpose(oihw_kernel, axes: [2, 3, 1, 0])

      run(input, hwio_kernel, kernel_permutation: [3, 2, 0, 1])
    end

    test "output_permutation: [0, 2, 3, 1] (caller wants NHWC output)" do
      {input, kernel} = inputs({1, 3, 5, 5}, {4, 3, 3, 3})
      run(input, kernel, output_permutation: [0, 2, 3, 1])
    end

    test "all three permutations non-default (NHWC end-to-end)" do
      {nchw_input, oihw_kernel} = inputs({1, 3, 5, 5}, {4, 3, 3, 3})
      nhwc_input = Nx.transpose(nchw_input, axes: [0, 2, 3, 1])
      hwio_kernel = Nx.transpose(oihw_kernel, axes: [2, 3, 1, 0])

      run(nhwc_input, hwio_kernel,
        input_permutation: [0, 3, 1, 2],
        kernel_permutation: [3, 2, 0, 1],
        output_permutation: [0, 2, 3, 1]
      )
    end
  end

  describe "dtype coercion" do
    test "integer input is cast to float output dtype" do
      # Nx.conv produces float output even for integer operands; backend
      # must cast before dispatching to MLX (which is float-only for conv).
      input = Nx.iota({1, 2, 4, 4}, type: {:s, 32}, backend: Emily.Backend)
      kernel = Nx.iota({3, 2, 2, 2}, type: {:s, 32}, backend: Emily.Backend)

      ref_input = Nx.iota({1, 2, 4, 4}, type: {:s, 32}, backend: Nx.BinaryBackend)
      ref_kernel = Nx.iota({3, 2, 2, 2}, type: {:s, 32}, backend: Nx.BinaryBackend)

      emily_result = Nx.conv(input, kernel)
      ref_result = Nx.conv(ref_input, ref_kernel)

      assert_close(emily_result, ref_result)
    end
  end
end
