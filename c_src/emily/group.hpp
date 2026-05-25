// DistGroup: opaque resource wrapping mlx::core::distributed::Group.
//
// MLX's Group is itself a thin value type holding a shared_ptr to the
// backend's GroupImpl, so this wrapper just gives `fine` a concrete
// resource type to manage (mirroring Tensor in tensor.hpp). The group
// is the handle returned by `distributed::init` and threaded into every
// collective (all_sum, all_gather, send/recv, ...).

#pragma once

#include <fine.hpp>
#include <mlx/distributed/distributed.h>

#include <utility>

namespace emily {

class DistGroup {
public:
  DistGroup(mlx::core::distributed::Group group) : group(std::move(group)) {}
  mlx::core::distributed::Group group;
};

} // namespace emily
