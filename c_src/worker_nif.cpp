// Worker-thread NIFs: create / stop / introspect WorkerThread resources.
//
// The reaper that joins worker threads off the BEAM schedulers is shut
// down in the NIF unload callback registered below (see emily/worker.hpp).

#include "emily/worker.hpp"

#include <fine.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

using emily::Reaper;
using emily::WorkerThread;

FINE_RESOURCE(WorkerThread);

namespace {

// Join every worker thread when the NIF library is unloaded, so no worker
// is left running native code after the library goes away.
fine::Registration worker_unload_registration =
    fine::Registration::register_unload(
        [](ErlNifEnv *, void *) { Reaper::instance().shutdown(); });

fine::ResourcePtr<WorkerThread> create_worker(ErlNifEnv *, int64_t queue_limit) {
  if (queue_limit < 1) {
    throw std::invalid_argument("worker queue limit must be positive");
  }
  return fine::make_resource<WorkerThread>(
      static_cast<std::size_t>(queue_limit));
}
FINE_NIF(create_worker, 0);

// Stop a worker: cancel its queued tasks ({:error, :stopped}) and let it
// finish the in-flight op. Non-blocking — the reaper joins the thread when
// the resource is collected.
fine::Ok<> stop_worker(ErlNifEnv *, fine::ResourcePtr<WorkerThread> w) {
  w->stop();
  return fine::Ok<>{};
}
FINE_NIF(stop_worker, 0);

int64_t worker_queue_depth(ErlNifEnv *, fine::ResourcePtr<WorkerThread> w) {
  return static_cast<int64_t>(w->queue_depth());
}
FINE_NIF(worker_queue_depth, 0);

} // namespace
