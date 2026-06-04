defmodule Emily.Program do
  @moduledoc false
  # Thin Elixir wrapper over the `compile_program` / `eval_program`
  # NIFs (c_src/program.cpp). Compiles an `%Emily.IR{}` into a
  # replayable program resource and runs it on a worker, returning
  # native tensor refs.
  #
  # This is the substrate for the Expr compiler: `Emily.Compiler` will
  # lower a traced `Nx.Defn.Expr` to an `%Emily.IR{}`, compile it once
  # (cached in the closure), and replay it per call. CM0 exercises it
  # directly from tests to prove the dispatch-collapse thesis before the
  # full lowerer lands in CM1.

  alias Emily.{IR, Native}

  @type t :: reference()

  @doc """
  Compile an `%Emily.IR{}` into a replayable program resource.

  Captures and consts are native tensor refs held by the resource for
  its lifetime — weights cross the NIF boundary once here, never per
  eval. Raises `ArgumentError` on a malformed IR (unknown opcode,
  out-of-range or forward/cyclic operand ref, opcode/operand length
  mismatch).
  """
  @spec compile(IR.t()) :: t()
  def compile(%IR{} = ir) do
    opcodes = Enum.map(ir.instrs, fn %{opcode: op} -> IR.opcode(op) end)
    operands = Enum.map(ir.instrs, fn %{operands: ops} -> Enum.map(ops, &IR.pack_ref/1) end)
    outputs = Enum.map(ir.outputs, &IR.pack_ref/1)

    Native.compile_program(ir.n_inputs, ir.captures, ir.consts, opcodes, operands, outputs)
  end

  @doc """
  Replay `program` with `inputs` (native tensor refs, in slot order) on
  `worker`, returning the output tensor refs.

  Options:

    * `:mode` — how the output roots are evaluated after the DAG is
      built in C++:
        * `:sync` (default) — `mx::eval`: block on the GPU before
          returning.
        * `:async` — `mx::async_eval`: return as soon as the work is
          enqueued, for an overlapped decode loop.
        * `:build` — no eval: return the lazy graph. Isolates the
          build/dispatch cost and lets a caller `async_eval` several
          programs together.
  """
  @spec eval(Native.worker(), t(), [Native.tensor()], keyword()) :: [Native.tensor()]
  def eval(worker, program, inputs, opts \\ []) do
    eval_mode =
      case Keyword.get(opts, :mode, :sync) do
        :sync ->
          0

        :async ->
          1

        :build ->
          2

        other ->
          raise ArgumentError,
                "Emily.Program.eval/4 :mode must be :sync, :async or :build, got #{inspect(other)}"
      end

    Native.eval_program(worker, program, inputs, eval_mode)
  end

  @doc """
  Reflect a compiled program's stored IR back as
  `{n_inputs, n_captures, n_consts, opcodes, operands, outputs}`, for
  round-trip tests.
  """
  @spec describe(t()) ::
          {non_neg_integer(), non_neg_integer(), non_neg_integer(), [integer()], [[integer()]],
           [integer()]}
  def describe(program), do: Native.describe_program(program)
end
