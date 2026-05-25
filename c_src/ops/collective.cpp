// Distributed collectives over MLX's communication backends
// (`ring` = TCP, `jaccl` = RDMA-over-Thunderbolt).
//
// Two flavours of NIF live here:
//
//   * Control plane — is_available / init / rank / size. `init` forms
//     the inter-process connections (TCP accept/connect for ring), which
//     can block until every peer arrives, so it is dispatched onto the
//     worker thread via async_reply just like a compute op; rank/size
//     are cheap synchronous reads off the resulting Group.
//
//   * Collectives — all_sum/all_max/all_min/all_gather/sum_scatter and
//     send/recv. These are lazy MLX primitives: they enroll a node in
//     the graph and the actual communication happens during mx::eval on
//     the worker stream. Mechanically identical to the elementwise ops
//     in binary.cpp — dispatched with async_encoded, returning a Tensor.
//
// One rank == one OS process. A single BEAM node is one rank; multi-rank
// means multiple nodes (see Emily.Distributed.Launcher).

#include "../emily/async.hpp"
#include "../emily/group.hpp"
#include "../emily/tensor.hpp"
#include "../emily/worker.hpp"

#include <fine.hpp>
#include <mlx/distributed/distributed.h>
#include <mlx/distributed/ops.h>
#include <mlx/mlx.h>

#include <string>
#include <utility>
#include <vector>

namespace mx = mlx::core;
namespace dist = mlx::core::distributed;
using emily::async_encoded;
using emily::DistGroup;
using emily::Tensor;
using emily::to_mlx_dtype;
using emily::to_mlx_shape;
using emily::wrap;
using emily::WorkerThread;

FINE_RESOURCE(DistGroup);

namespace {

// ---------- Control plane ----------

// is_available/0 — is any communication backend usable in this build?
bool distributed_available(ErlNifEnv *) { return dist::is_available(); }
FINE_NIF(distributed_available, 0);

// is_available/1 — is the named backend ("ring" | "jaccl" | "mpi" |
// "nccl") usable here? jaccl reports false without cabled Thunderbolt
// RDMA hardware; ring is available wherever TCP is.
bool distributed_available_backend(ErlNifEnv *, std::string backend) {
  return dist::is_available(backend);
}
FINE_NIF(distributed_available_backend, 0);

// init/3 — initialise the distributed subsystem and return the world
// Group. Dispatched on the worker because the ring backend blocks here
// on the peer handshake. With strict=false and no launch environment
// MLX returns a singleton group (size 1) and collectives become no-ops.
fine::Term distributed_init_nif(ErlNifEnv *env,
                                fine::ResourcePtr<WorkerThread> w,
                                bool strict,
                                std::string backend) {
  return async_encoded(
      env, w, [strict, backend = std::move(backend)](mx::Stream &) {
        return fine::make_resource<DistGroup>(dist::init(strict, backend));
      });
}
FINE_NIF(distributed_init_nif, 0);

// rank/1, size/1 — cheap reads off the Group; no worker needed.
int64_t group_rank(ErlNifEnv *, fine::ResourcePtr<DistGroup> g) {
  return g->group.rank();
}
FINE_NIF(group_rank, 0);

int64_t group_size(ErlNifEnv *, fine::ResourcePtr<DistGroup> g) {
  return g->group.size();
}
FINE_NIF(group_size, 0);

// ---------- Collectives ----------

// Each takes (worker, x, group) and returns a Tensor, exactly like the
// binary ops. The communication is deferred to eval on the worker
// stream; here we only build the graph node.
#define EMILY_COLLECTIVE(op_name, mlx_fn)                                      \
  fine::Term op_name##_nif(                                                    \
      ErlNifEnv *env,                                                          \
      fine::ResourcePtr<WorkerThread> w,                                       \
      fine::ResourcePtr<Tensor> x,                                             \
      fine::ResourcePtr<DistGroup> g) {                                        \
    return async_encoded(env, w,                                               \
        [x = std::move(x), g = std::move(g)](mx::Stream &s) {                  \
          return wrap(mlx_fn(x->array, g->group, s));                          \
        });                                                                    \
  }                                                                            \
  FINE_NIF(op_name##_nif, 0);

EMILY_COLLECTIVE(dist_all_sum, dist::all_sum)
EMILY_COLLECTIVE(dist_all_max, dist::all_max)
EMILY_COLLECTIVE(dist_all_min, dist::all_min)
EMILY_COLLECTIVE(dist_all_gather, dist::all_gather)
EMILY_COLLECTIVE(dist_sum_scatter, dist::sum_scatter)

#undef EMILY_COLLECTIVE

// send/3 — send x to rank `dst`; returns x (for graph ordering).
fine::Term dist_send_nif(ErlNifEnv *env,
                         fine::ResourcePtr<WorkerThread> w,
                         fine::ResourcePtr<Tensor> x,
                         int64_t dst,
                         fine::ResourcePtr<DistGroup> g) {
  return async_encoded(
      env, w, [x = std::move(x), dst, g = std::move(g)](mx::Stream &s) {
        return wrap(dist::send(x->array, static_cast<int>(dst), g->group, s));
      });
}
FINE_NIF(dist_send_nif, 0);

// recv/4 — receive an array of (shape, dtype) from rank `src`.
fine::Term dist_recv_nif(ErlNifEnv *env,
                         fine::ResourcePtr<WorkerThread> w,
                         std::vector<int64_t> shape,
                         std::tuple<fine::Atom, int64_t> dtype_tuple,
                         int64_t src,
                         fine::ResourcePtr<DistGroup> g) {
  return async_encoded(
      env, w,
      [shape = std::move(shape), dtype_tuple, src,
       g = std::move(g)](mx::Stream &s) {
        return wrap(dist::recv(to_mlx_shape(shape), to_mlx_dtype(dtype_tuple),
                               static_cast<int>(src), g->group, s));
      });
}
FINE_NIF(dist_recv_nif, 0);

} // namespace
