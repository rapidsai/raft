#pragma once
#include <raft/core/detail/device_id_base.hpp>
#include <raft/core/device_type.hpp>

namespace raft {
namespace detail {
template <>
struct device_id<device_type::cpu> {
  using value_type = int;
  device_id() noexcept : id_{value_type{}} {}
  device_id(value_type dev_id) noexcept : id_{dev_id} {}

  auto value() const noexcept { return id_; }
  auto rmm_id() const {
    throw bad_device_type{"CPU devices have no RMM ID"};
  }
 private:
  value_type id_;
};
}  // namespace detail
}  // namespace raft
