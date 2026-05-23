defmodule Emily.Native.Async do
  @moduledoc false
  # Helper for awaiting async NIF replies.
  #
  # Async NIFs return a ref synchronously and dispatch the actual
  # work onto a worker thread that posts
  # `{ref, {:ok, result}}` or `{ref, {:error, reason}}` back to the
  # caller PID via `enif_send`. `call/2` awaits that message and, if
  # given a `context` map, appends operation/input/option diagnostics
  # to the raised error.
  #
  # Error reasons mirror `fine::nif_impl`'s sync catch ladder:
  #
  #   {:argument, binary} -> ArgumentError
  #   {:runtime, binary}  -> RuntimeError
  #   :unknown            -> RuntimeError
  #
  # The error-formatting path is intentionally total: any unexpected
  # context shape (nil keys, non-list inputs, non-stringifiable op)
  # degrades to a "context=?" marker rather than crashing inside the
  # raise and masking the underlying NIF failure.
  #
  # See `c_src/emily/async.hpp` and
  # `docs/planning/async-worker-exploration.md`.

  @type tensor_context ::
          reference()
          | %{optional(:shape) => term(), optional(:dtype) => term()}
          | nil

  @type context :: %{
          optional(:op) => atom() | String.t(),
          optional(:inputs) => keyword(tensor_context()) | [tensor_context()],
          optional(:options) => keyword(),
          optional(:stream) => term()
        }

  @inspect_opts [limit: 50, printable_limit: 1024, charlists: :as_lists]

  @doc """
  Await the reply posted by the worker thread for `ref`.

  Blocks the calling process on `receive/1`, not any BEAM scheduler
  — the scheduler can run other work while the worker executes the
  op.
  """
  @spec call(reference(), context() | nil) :: term()
  def call(ref, context \\ nil) do
    receive do
      {^ref, {:ok, result}} ->
        result

      {^ref, {:error, {:argument, message}}} ->
        raise ArgumentError, add_context(message, context)

      {^ref, {:error, {:runtime, message}}} ->
        raise RuntimeError, add_context(message, context)

      {^ref, {:error, :unknown}} ->
        raise RuntimeError, add_context("unknown exception thrown within NIF", context)
    end
  end

  defp add_context(message, nil), do: message

  defp add_context(message, context) when is_map(context) do
    details =
      [
        op_context(context),
        input_context(Map.get(context, :inputs)),
        option_context(Map.get(context, :options)),
        stream_context(Map.get(context, :stream))
      ]
      |> Enum.reject(&(&1 in [nil, ""]))
      |> Enum.join(" ")

    if details == "", do: message, else: message <> "\nEmily.Native context: " <> details
  end

  defp add_context(message, _other), do: message

  defp op_context(%{op: op}) do
    "op=" <> safe_to_string(op)
  end

  defp op_context(_context), do: nil

  defp input_context(nil), do: nil
  defp input_context([]), do: nil

  defp input_context(inputs) when is_list(inputs) do
    rendered =
      inputs
      |> Enum.with_index()
      |> Enum.map_join(", ", &render_input/1)

    "inputs=[" <> rendered <> "]"
  end

  defp input_context(_other), do: "inputs=?"

  defp render_input({{name, tensor}, _index}) when is_atom(name) do
    "#{name}: #{tensor_context(tensor)}"
  end

  defp render_input({tensor, index}) do
    "#{index}: #{tensor_context(tensor)}"
  end

  defp option_context(nil), do: nil
  defp option_context([]), do: nil

  defp option_context(options) when is_list(options) do
    "options=" <> inspect(options, @inspect_opts)
  end

  defp option_context(_other), do: "options=?"

  defp stream_context(nil), do: nil
  defp stream_context(stream), do: "stream=" <> inspect(stream, @inspect_opts)

  defp tensor_context(nil), do: "nil"

  defp tensor_context(%{shape: shape, dtype: dtype}) do
    format_metadata(shape, dtype)
  end

  defp tensor_context(%{shape: shape, type: type}) do
    format_metadata(shape, type)
  end

  defp tensor_context(%{shape: shape}) do
    "shape=" <> inspect(shape, @inspect_opts) <> " dtype=?"
  end

  defp tensor_context(tensor) when is_reference(tensor) do
    case tensor_metadata(tensor) do
      {:ok, shape, dtype} -> format_metadata(shape, dtype)
      :error -> "shape=? dtype=?"
    end
  end

  defp tensor_context(other), do: safe_inspect(other)

  defp format_metadata(shape, dtype) do
    "shape=" <> inspect(shape, @inspect_opts) <> " dtype=" <> inspect(dtype, @inspect_opts)
  end

  defp tensor_metadata(tensor) do
    {:ok, Emily.Native.shape(tensor), Emily.Native.dtype(tensor)}
  rescue
    _ -> :error
  catch
    _, _ -> :error
  end

  defp safe_to_string(value) do
    to_string(value)
  rescue
    _ -> inspect(value, @inspect_opts)
  catch
    _, _ -> inspect(value, @inspect_opts)
  end

  defp safe_inspect(value) do
    inspect(value, @inspect_opts)
  rescue
    _ -> "?"
  catch
    _, _ -> "?"
  end
end
