#pragma once
#include <raft/core/device_type.hpp>

namespace raft {
namespace detail {
template<device_type D>
struct device_id {
  using value_type = int;

  device_id(value_type device_index) {}
  auto value() const { return value_type{}; }
};
}  // namespace detail
}  // namespace raft
