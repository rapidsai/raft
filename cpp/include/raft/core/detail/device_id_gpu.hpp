#pragma once
#include <raft/util/cudart_utils.hpp>
#include <raft/core/detail/device_id_base.hpp>
#include <raft/core/device_type.hpp>
#include <rmm/cuda_device.hpp>

namespace raft {
namespace detail {
template<>
struct device_id<device_type::gpu> {
  using value_type = typename rmm::cuda_device_id::value_type;
  device_id() noexcept(false) : id_{[](){
    auto raw_id = value_type{};
    kayak::cuda_check(cudaGetDevice(&raw_id));
    return raw_id;
  }()} {};
  device_id(value_type dev_id) noexcept : id_{dev_id} {};

  auto value() const noexcept { return id_.value(); }
  auto rmm_id() const noexcept { return id_; }
 private:
  rmm::cuda_device_id id_;
};
}  // namespace detail
}  // namespace raft

