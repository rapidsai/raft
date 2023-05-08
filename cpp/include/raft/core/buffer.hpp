/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "raft/core/logger.hpp"
#include <cstddef>
#include <iterator>
#include <memory>
#include <raft/core/detail/buffer_utils/buffer_copy.hpp>
#include <raft/core/detail/buffer_utils/non_owning_buffer.hpp>
#include <raft/core/detail/buffer_utils/owning_buffer.hpp>
#include <raft/core/detail/const_agnostic.hpp>
#include <raft/core/device_support.hpp>
#include <raft/core/device_type.hpp>
#include <raft/core/error.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resources.hpp>
#include <stdint.h>
#include <utility>
#include <variant>

namespace raft {
/**
 * @brief A container which may or may not own its own data on host or device
 *
 */
template <typename ElementType,
          typename LayoutPolicy,
          template <typename T>
          typename ContainerPolicy,
          typename IndexType = std::uint32_t>
struct buffer {
  using data_store = std::variant<detail::non_owning_buffer<ElementType, device_type::cpu, IndexType, LayoutPolicy, ContainerPolicy>,
                                  detail::non_owning_buffer<ElementType, device_type::gpu, IndexType, LayoutPolicy, ContainerPolicy>,
                                  detail::owning_buffer<ElementType, device_type::cpu, IndexType, LayoutPolicy, ContainerPolicy>,
                                  detail::owning_buffer<ElementType, device_type::gpu, IndexType, LayoutPolicy, ContainerPolicy>>;

  buffer() : device_type_{}, data_{}, size_{}, memory_type_{memory_type::host} {}

  /** Construct non-initialized owning buffer */
  buffer(raft::resources const& handle,
         IndexType size,
         memory_type mem_type    = memory_type::host)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      data_{[this, mem_type, size, handle]() {
        auto result = data_store{};
        if (is_device_accessible(mem_type)) {
          result = detail::owning_buffer<ElementType, device_type::gpu, IndexType, LayoutPolicy, ContainerPolicy>{handle, size};
        } else {
          result = detail::owning_buffer<ElementType, device_type::cpu, IndexType, LayoutPolicy, ContainerPolicy>{handle, size};
        }
        return result;
      }()},
      size_{size},
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<ElementType*>(nullptr);
        switch (data_.index()) {
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
        }
        return result;
      }()}
  {
  }

  /** Construct non-owning buffer. Currently, users must ensure that the input_data is on the same device_type as the requested mem_type.
      This cannot be asserted because checking the device id requires cuda headers (which is against the intended cuda-free build). If
      the mem_type is different from the device_type of input_data, the input_data should first be copied to the appropriate location. */
  buffer(raft::resources const& handle, ElementType* input_data, IndexType size, memory_type mem_type = memory_type::host)
    : device_type_{[mem_type]() {
        RAFT_LOG_INFO("Non owning constructor call started");
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      data_{[this, input_data, mem_type]() {
        auto result = data_store{};
        if (is_device_accessible(mem_type)) {
          result = detail::non_owning_buffer<ElementType, device_type::gpu, IndexType, LayoutPolicy, ContainerPolicy>{input_data};
        } else {
          result = detail::non_owning_buffer<ElementType, device_type::cpu, IndexType, LayoutPolicy, ContainerPolicy>{input_data};
        }
        return result;
      }()},
      size_{size},
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<ElementType*>(nullptr);
        RAFT_LOG_INFO("data_index from constructor %d\n", data_.index());
        switch (data_.index()) {
          case 0: result = std::get<0>(data_).get(); break;
          case 1: result = std::get<1>(data_).get(); break;
        }
        RAFT_LOG_INFO("data pointer from constructor %p\n", result);
        return result;
      }()}
  {
    RAFT_LOG_INFO("Non owning constructor call complete");
  }

  /**
   * @brief Construct one buffer of the given memory type from another.
   * A buffer constructed in this way is owning and will copy the data from
   * the original location.
   */
  buffer(raft::resources const& handle,
         buffer const& other,
         memory_type mem_type)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      data_{[this, &other, mem_type, handle]() {
        auto result      = data_store{};
        auto result_data = static_cast<ElementType*>(nullptr);
        if (is_device_accessible(mem_type)) {
          auto buf =
            detail::owning_buffer<device_type::gpu, T>(handle, other.size());
          result_data = buf.get();
          result      = std::move(buf);
          RAFT_LOG_INFO("gpu copy called");
          detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::gpu, other.dev_type());   
        } else {
          auto buf    = detail::owning_buffer<device_type::cpu, T>(other.size());
          result_data = buf.get();
          result      = std::move(buf);
          RAFT_LOG_INFO("cpu copy called");
          detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::cpu, other.dev_type());
        }
        return result;
      }()},
      size_{other.size()},
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<T*>(nullptr);
        switch (data_.index()) {
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
        }
        return result;
      }()}
  {
    RAFT_LOG_INFO("Pointer to other's data %p\n", other.data_handle());
  }

  friend void swap(buffer<T>& first, buffer<T>& second)
  {
    using std::swap;
    swap(first.device_type_, second.device_type_);
    swap(first.data_, second.data_);
    swap(first.size_, second.size_);
    swap(first.memory_type_, second.memory_type_);
    swap(first.cached_ptr, second.cached_ptr);
  }
  buffer<T>& operator=(buffer<T> const& other) {
    auto copy = other;
    swap(*this, copy);
    return *this;
  }

  /**
   * @brief Create owning copy of existing buffer with given stream
   * The device type of this new buffer will be the same as the original
   */
  buffer(raft::resources const& handle, buffer<T> const& other) : buffer(handle, other, other.mem_type())
  {
  }

  /**
   * @brief Move from existing buffer unless a copy is necessary based on
   * memory location
   */
  buffer(raft::resources const& handle, buffer<T>&& other, memory_type mem_type)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      data_{[&other, mem_type, handle]() {
        auto result = data_store{};
        if (mem_type == other.mem_type()) {
          result = std::move(other.data_);
        } else {
          auto* result_data = static_cast<T*>(nullptr);
          if (is_device_accessible(mem_type)) {
            auto buf    = detail::owning_buffer<device_type::gpu, T>{handle, other.size()};
            auto buf = detail::owning_buffer<T,
                  device_type::gpu,
                  IndexType,
          typename LayoutPolicy = layout_c_contiguous,
          template <typename> typename ContainerPolicy>
            result_data = buf.get();
            result      = std::move(buf);
            detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::gpu, other.dev_type());
          } else {
            auto buf    = detail::owning_buffer<device_type::cpu, T>{other.size()};
            result_data = buf.get();
            result      = std::move(buf);
            detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::cpu, other.dev_type());
          }
        }
        return result;
      }()},
      size_{other.size()},
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<T*>(nullptr);
        switch (data_.index()) {
          case 0: result = std::get<0>(data_).get(); break;
          case 1: result = std::get<1>(data_).get(); break;
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
        }
        return result;
      }()}
  {
    RAFT_LOG_INFO("main move called");
  }
  // buffer(buffer<T>&& other, device_type mem_type)
  //   : buffer{std::move(other), mem_type, 0, execution_stream{}}
  // {
  //   RAFT_LOG_INFO("copy constructor without stream and device called");
  // }

  buffer(buffer<T>&& other) noexcept
    : device_type_{[&other]() {
        return is_device_accessible(other.mem_type()) ? device_type::gpu : device_type::cpu;
      }()},
      data_{[&other]() {
        auto result = data_store{};
          result = std::move(other.data_);
          return result;
      }()},
      size_{other.size()},
      memory_type_{other.mem_type()},
      cached_ptr{[this]() {
        auto result = static_cast<T*>(nullptr);
        switch (data_.index()) {
          case 0: result = std::get<0>(data_).get(); break;
          case 1: result = std::get<1>(data_).get(); break;
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
        }
        return result;
      }()}
  {
    RAFT_LOG_INFO("trivial move called");
  }
  buffer<T>& operator=(buffer<T>&& other) noexcept {
    RAFT_LOG_INFO("operator= move called");
    data_ = std::move(other.data_);
    device_type_ = std::move(other.device_type_);
    size_ = std::move(other.size_);
    memory_type_ = std::move(other.memory_type_);
    cached_ptr = std::move(other.cached_ptr);
    return *this;
  }

  auto size() const noexcept { return size_; }
  HOST DEVICE auto* data_handle() const noexcept { 
    auto result = static_cast<T*>(nullptr);
    switch (data_.index()) {
      case 0: result = std::get<0>(data_).get(); break;
      case 1: result = std::get<1>(data_).get(); break;
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
    }
      RAFT_LOG_INFO("data_handle() called: data %p; cached_ptr %p\n", result, cached_ptr);
        return result;}

  auto device() const noexcept { return device_type_; }

  auto mem_type() const noexcept
  {
    return memory_type_;
  }

  ~buffer() = default;

  auto view() -> view_type {
  return make_mdspan<ElementType, IndexType, LayoutPolicy, is_host_accessible(this -> mem_type()), is_device_accessible(this -> mem_type())>(data_, make_extents<IndexType>(size_));
  }

 private:
 auto dev_type() const noexcept
  {
    return device_type_;
  }

  enum device_type device_type_;
  data_store data_;
  IndexType size_;
  enum memory_type memory_type_;
  ElementType* cached_ptr;
  int device_id_;
};

template <bool bounds_check, typename T, typename U>
detail::const_agnostic_same_t<T, U> copy(raft::resources const& handle,
                                         buffer<T>& dst,
                                         buffer<U> const& src,
                                         size_t dst_offset,
                                         size_t src_offset,
                                         size_t size)
{
  if constexpr (bounds_check) {
    if (src.size() - src_offset < size || dst.size() - dst_offset < size) {
      throw out_of_bounds("Attempted copy to or from buffer of inadequate size");
    }
  }
  auto src_device_type = is_device_accessible(src.mem_type()) ? device_type::gpu : device_type::cpu;
  auto dst_device_type = is_device_accessible(dst.mem_type()) ? device_type::gpu : device_type::cpu;
  detail::buffer_copy(handle,
                      dst.data_handle() + dst_offset,
                      src.data_handle() + src_offset,
                      size,
                      dst_device_type,
                      src_device_type);
}

template <bool bounds_check, typename T, typename U>
detail::const_agnostic_same_t<T, U> copy(raft::resources const& handle,
                                         buffer<T>& dst,
                                         buffer<U> const& src)
{
  copy<bounds_check>(handle, dst, src, 0, 0, src.size());
}
}  // namespace raft