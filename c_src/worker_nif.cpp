// Worker-thread NIFs: create and manage WorkerThread resources.

#include "emily/worker.hpp"

#include <fine.hpp>

using emily::WorkerThread;

FINE_RESOURCE(WorkerThread);

namespace {

fine::ResourcePtr<WorkerThread> create_worker(ErlNifEnv *) {
  return fine::make_resource<WorkerThread>();
}
FINE_NIF(create_worker, 0);

} // namespace
