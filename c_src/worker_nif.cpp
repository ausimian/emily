// Worker-thread NIFs: create and manage WorkerThread resources.

#include "emily/worker.hpp"

#include <fine.hpp>
#include <mlx/mlx.h>

namespace mx = mlx::core;
using emily::WorkerThread;

FINE_RESOURCE(WorkerThread);

namespace {

// GPU worker — the default for ordinary Nx/Emily compute.
fine::ResourcePtr<WorkerThread> create_worker(ErlNifEnv *) {
  return fine::make_resource<WorkerThread>();
}
FINE_NIF(create_worker, 0);

// CPU worker — for distributed collectives, which are CPU-only and
// would otherwise block the shared GPU worker during their eval.
fine::ResourcePtr<WorkerThread> create_cpu_worker(ErlNifEnv *) {
  return fine::make_resource<WorkerThread>(mx::Device::DeviceType::cpu);
}
FINE_NIF(create_cpu_worker, 0);

} // namespace
