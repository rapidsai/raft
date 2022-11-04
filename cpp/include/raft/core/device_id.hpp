#pragma once

#include <raft/core/detail/device_id_base.hpp>
#include <raft/core/detail/device_id_cpu.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/detail/device_id_gpu.hpp>
#endif
#include <raft/core/device_type.hpp>
#include <variant>

namespace raft {
template <device_type D>
using device_id = detail::device_id<D>;

using device_id_variant = std::variant<device_id<device_type::cpu>, device_id<device_type::gpu>>;
}
