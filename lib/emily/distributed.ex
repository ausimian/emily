defmodule Emily.Distributed do
  @moduledoc """
  Multi-process collectives over MLX's communication backends.

  MLX distributes work across **processes**, one rank per OS process —
  so one BEAM node is one rank, and a multi-rank job is multiple nodes.
  The collectives here (`all_sum/2`, `all_gather/2`, ...) are the same
  primitives MLX uses internally; the heavy tensor traffic flows over
  MLX's own transport inside the NIF, *not* over Erlang distribution.

  ## Backends

    * `"ring"` — TCP ring; works over any network, including loopback,
      so it is fully exercisable on a single machine. The default for
      development.
    * `"jaccl"` — RDMA over Thunderbolt; only available on macOS with
      cabled Apple Silicon machines (and SDK ≥ 26.2). `available?/1`
      reports `false` otherwise.

  ## Single-node development

  Use `Emily.Distributed.Launcher` to spin up N local ranks as BEAM
  peer nodes over loopback — the same orchestration that extends to
  multiple Macs by swapping the peer `exec` for ssh.

  ## Bootstrap

  Each rank discovers its peers from the environment MLX reads at
  `init/1`: `MLX_RANK` (this rank) and `MLX_HOSTFILE` (a JSON list, in
  rank order, of `"ip:port"` addresses). The launcher sets these; if
  you launch ranks yourself, set them before calling `init/1`. With
  neither set and `strict: false`, `init/1` returns a singleton group
  (`size == 1`) and every collective is a no-op — handy for code that
  should run identically clustered or not.

  ## Status

  Scaffold. The native binding and API shape are in place; production
  concerns (fault handling when a rank dies, `Group.split/2`,
  integration with `Nx.Defn` sharding) are not yet addressed.
  """

  alias Emily.Backend
  alias Emily.Native
  alias Nx.Tensor, as: T

  defmodule Group do
    @moduledoc "Handle to an initialised distributed world (one per node)."
    @enforce_keys [:ref, :worker, :rank, :size, :backend]
    defstruct [:ref, :worker, :rank, :size, :backend]

    @type t :: %__MODULE__{
            ref: reference(),
            worker: reference(),
            rank: non_neg_integer(),
            size: pos_integer(),
            backend: String.t()
          }
  end

  @doc """
  Whether a communication backend is available in this build.

  Pass a backend name (`"ring"`, `"jaccl"`, ...) to probe a specific
  one, or omit it to ask whether *any* backend is usable.
  """
  @spec available?(String.t() | nil) :: boolean()
  def available?(backend \\ nil)
  def available?(nil), do: Native.distributed_available()

  def available?(backend) when is_binary(backend),
    do: Native.distributed_available_backend(backend)

  @doc """
  Initialise the distributed subsystem and return the world `Group`.

  ## Options

    * `:backend` — `"any"` (default), `"ring"`, `"jaccl"`, ...
    * `:strict` — when `true`, raise if the backend can't initialise
      instead of falling back to a singleton group. Default `false`.
    * `:worker` — worker (MLX stream) the collectives run on. Defaults
      to the backend's shared worker so collective results live on the
      same stream as the tensors they operate on; MLX streams are
      thread-local, so a result produced on one stream can't be read
      back on another. The group handle itself is process-global, so it
      is independent of the worker.
  """
  @spec init(keyword()) :: Group.t()
  def init(opts \\ []) do
    backend = Keyword.get(opts, :backend, "any")
    strict = Keyword.get(opts, :strict, false)
    worker = Keyword.get_lazy(opts, :worker, &Emily.MlxStream.default_worker/0)

    ref = Native.distributed_init(worker, strict, backend)

    %Group{
      ref: ref,
      worker: worker,
      rank: Native.group_rank(ref),
      size: Native.group_size(ref),
      backend: backend
    }
  end

  @doc "This process's rank within the group (0-based)."
  @spec rank(Group.t()) :: non_neg_integer()
  def rank(%Group{rank: r}), do: r

  @doc "Number of ranks in the group."
  @spec size(Group.t()) :: pos_integer()
  def size(%Group{size: s}), do: s

  @doc "All-reduce by sum: every rank receives the elementwise sum across ranks."
  @spec all_sum(Group.t(), T.t()) :: T.t()
  def all_sum(group, tensor), do: collective(:dist_all_sum, group, tensor)

  @doc "All-reduce by max."
  @spec all_max(Group.t(), T.t()) :: T.t()
  def all_max(group, tensor), do: collective(:dist_all_max, group, tensor)

  @doc "All-reduce by min."
  @spec all_min(Group.t(), T.t()) :: T.t()
  def all_min(group, tensor), do: collective(:dist_all_min, group, tensor)

  @doc "Gather each rank's tensor, concatenated along axis 0."
  @spec all_gather(Group.t(), T.t()) :: T.t()
  def all_gather(group, tensor), do: collective(:dist_all_gather, group, tensor)

  @doc "Reduce-scatter by sum: sum across ranks, each rank keeping its slice."
  @spec sum_scatter(Group.t(), T.t()) :: T.t()
  def sum_scatter(group, tensor), do: collective(:dist_sum_scatter, group, tensor)

  @doc "Send `tensor` to rank `dst`. Returns `tensor` (for graph ordering)."
  @spec send(Group.t(), T.t(), non_neg_integer()) :: T.t()
  def send(%Group{ref: gref, worker: w}, %T{} = tensor, dst) when is_integer(dst) do
    to_nx(Native.dist_send(w, ref(tensor), dst, gref))
  end

  @doc "Receive a tensor of `shape`/`type` from rank `src`."
  @spec recv(Group.t(), tuple(), Nx.Type.t(), non_neg_integer()) :: T.t()
  def recv(%Group{ref: gref, worker: w}, shape, type, src)
      when is_tuple(shape) and is_integer(src) do
    to_nx(Native.dist_recv(w, Tuple.to_list(shape), type, src, gref))
  end

  # --- internals ---------------------------------------------------

  defp collective(op, %Group{ref: gref, worker: w}, %T{} = tensor) do
    apply(Native, op, [w, ref(tensor), gref]) |> to_nx()
  end

  defp ref(%T{data: %Backend{ref: r}}), do: r
  defp ref(%T{} = other), do: ref(Nx.backend_transfer(other, Backend))

  # Rebuild an Nx tensor around a native result ref. We read shape/dtype
  # back from the device rather than recomputing the collective's shape
  # math (all_gather/sum_scatter change axis 0).
  defp to_nx(result_ref) do
    type = Native.dtype(result_ref)
    shape = result_ref |> Native.shape() |> List.to_tuple()
    names = List.duplicate(nil, tuple_size(shape))
    %T{shape: shape, type: type, names: names, data: %Backend{ref: result_ref}}
  end
end
