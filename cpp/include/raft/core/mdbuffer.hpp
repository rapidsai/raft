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
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <raft/core/detail/buffer_utils/buffer_copy.hpp>
#include <raft/core/buffer_container_policy.hpp>
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
 * @tparam ElementType type of the input
 * @tparam LayoutPolicy layout of the input
 * @tparam ContainerPolicy container to be used to own host/device memory if needed.
 * Users must ensure that the container has the correct type (host/device). Exceptions
 * due to a device container being used for a host mdbuffer and vice versa are not caught
 * by the mdbuffer class.
 * @tparam the index type of the extents
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy = layout_c_contiguous,
          template <typename> typename ContainerPolicy = buffer_container_policy>
struct mdbuffer {
  using data_store = std::variant<detail::non_owning_buffer<ElementType, memory_type::host, Extents, LayoutPolicy, ContainerPolicy>,
                                  detail::non_owning_buffer<ElementType, memory_type::device, Extents, LayoutPolicy, ContainerPolicy>,
                                  detail::non_owning_buffer<ElementType, memory_type::managed, Extents, LayoutPolicy, ContainerPolicy>,   
                                  detail::owning_host_buffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>,
                                  detail::owning_device_buffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>>;

  mdbuffer() : device_type_{}, data_{}, length_{0}, memory_type_{memory_type::host} {}

  /** Construct non-initialized owning mdbuffer. For owning buffers, managed memory is treated as
   * device memory only. Therefore, users are discouraged from using managed memory for creating
   * owning buffers. */
  mdbuffer(raft::resources const& handle,
         Extents extents,
         memory_type mem_type = memory_type::host)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      extents_{extents},
      data_{[this, mem_type, handle]() {
        auto result = data_store{};
        if (is_device_accessible(mem_type)) {
          result = detail::owning_device_buffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>{handle, extents_};
        } else {
          result = detail::owning_host_buffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>{handle, extents_};
        }
        return result;
      }()},
      length_([this]() {
        size_t length = 1;
        for (size_t i = 0; i < extents_.rank(); ++i) {
          length *= extents_.extent(i);
        }
        return length;
      }()),
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<ElementType*>(nullptr);
        switch (data_.index()) {
          case 3: result = std::get<3>(data_).get(); break;
          case 4: result = std::get<4>(data_).get(); break;
        }
        return result;
      }()}
  {
  }

  /** Construct non-owning mdbuffer. Currently, users must ensure that the input_data is on the same device_type as the requested mem_type.
      This cannot be asserted because checking the device id requires CUDA headers (which is against the intended cpu-gpu interop). If
      the mem_type is different from the device_type of input_data, the input_data should first be copied to the appropriate location. For
      managed memory_type, input_data should be a managed pointer. */
  mdbuffer(raft::resources const& handle, ElementType* input_data, Extents extents, memory_type mem_type = memory_type::host)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      extents_{extents},
      data_{[this, input_data, mem_type]() {
        auto result = data_store{};
        if (is_host_device_accessible(mem_type)) {
          result = detail::non_owning_buffer<ElementType, memory_type::managed, Extents, LayoutPolicy, ContainerPolicy>{input_data, extents_};
        } else if (is_device_accessible(mem_type)) {
          result = detail::non_owning_buffer<ElementType, memory_type::device, Extents, LayoutPolicy, ContainerPolicy>{input_data, extents_};
        } else {
          result = detail::non_owning_buffer<ElementType, memory_type::host, Extents, LayoutPolicy, ContainerPolicy>{input_data, extents_};
        }
        return result;
      }()},
      length_([this]() {
        std::size_t length = 1;
        for (std::size_t i = 0; i < extents_.rank(); ++i) {
          length *= extents_.extent(i);
        }
        return length;
      }()),
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<ElementType*>(nullptr);
        switch (data_.index()) {
          case 0: result = std::get<0>(data_).get(); break;
          case 1: result = std::get<1>(data_).get(); break;
          case 2: result = std::get<1>(data_).get(); break;
        }
        return result;
      }()}
  {
  }

  /**
   * @brief Construct one mdbuffer of the given memory type from another.
   * A mdbuffer constructed in this way is owning and will copy the data from
   * the original location.
   */
  mdbuffer(raft::resources const& handle,
         mdbuffer const& other,
         memory_type mem_type)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      extents_{other.extents()},
      data_{[this, &other, mem_type, handle]() {
        auto result      = data_store{};
        auto result_data = static_cast<ElementType*>(nullptr);
        if (is_device_accessible(mem_type)) {
          auto buf =
            detail::owning_device_buffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>(handle, extents_);
          result_data = buf.get();
          result      = std::move(buf);
          detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::gpu, other.dev_type());   
        } else {
          auto buf    = detail::owning_host_buffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>(handle, extents_);
          result_data = buf.get();
          result      = std::move(buf);
          detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::cpu, other.dev_type());
        }
        return result;
      }()},
      length_([this]() {
        std::size_t length = 1;
        for (std::size_t i = 0; i < extents_.rank(); ++i) {
          length *= extents_.extent(i);
        }
        return length;
      }()),
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

  friend void swap(mdbuffer<ElementType, Extents>& first, mdbuffer<ElementType, Extents>& second)
  {
    using std::swap;
    swap(first.device_type_, second.device_type_);
    swap(first.data_, second.data_);
    swap(first.size_, second.size_);
    swap(first.memory_type_, second.memory_type_);
    swap(first.cached_ptr, second.cached_ptr);
  }
  mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>& operator=(mdbuffer<ElementType, Extents> const& other) {
    auto copy = other;
    swap(*this, copy);
    return *this;
  }

  /**
   * @brief Create owning copy of existing mdbuffer with given stream
   * The device type of this new mdbuffer will be the same as the original
   */
  mdbuffer(raft::resources const& handle, mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy> const& other) : mdbuffer(handle, other, other.mem_type())
  {
  }

  /**
   * @brief Move from existing mdbuffer unless a copy is necessary based on
   * memory location
   */
  mdbuffer(raft::resources const& handle, mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>&& other, memory_type mem_type)
    : device_type_{[mem_type]() {
        return is_device_accessible(mem_type) ? device_type::gpu : device_type::cpu;
      }()},
      extents_{other.extents()},
      data_{[&other, mem_type, handle, this]() {
        auto result = data_store{};
        if (mem_type == other.mem_type()) {
          result = std::move(other.data_);
        } else {
          auto* result_data = static_cast<ElementType*>(nullptr);
          if (is_device_accessible(mem_type)) {
            auto buf = detail::owning_device_buffer<ElementType,
                  Extents,
                  LayoutPolicy,
                  ContainerPolicy>{handle, extents_};
            result_data = buf.get();
            result      = std::move(buf);
            detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::gpu, other.dev_type());
          } else {
            auto buf = detail::owning_host_buffer<ElementType,
                  Extents, LayoutPolicy, ContainerPolicy>{handle, extents_};
            result_data = buf.get();
            result      = std::move(buf);
            detail::buffer_copy(handle, result_data, other.data_handle(), other.size(), device_type::cpu, other.dev_type());
          }
        }
        return result;
      }()},
      memory_type_{mem_type},
      cached_ptr{[this]() {
        auto result = static_cast<ElementType*>(nullptr);
        switch (data_.index()) {
          case 0: result = std::get<0>(data_).get(); break;
          case 1: result = std::get<1>(data_).get(); break;
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
          case 4: result = std::get<4>(data_).get(); break;
        }
        return result;
      }()}
  {
  }

  mdbuffer(mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>&& other) noexcept
    : device_type_{[&other]() {
        return is_device_accessible(other.mem_type()) ? device_type::gpu : device_type::cpu;
      }()},
      extents_{other.extents_},
      data_{[&other]() {
        auto result = data_store{};
          result = std::move(other.data_);
          return result;
      }()},
      length_{other.length_},
      memory_type_{other.mem_type()},
      cached_ptr{[this]() {
        auto result = static_cast<ElementType*>(nullptr);
        switch (data_.index()) {
          case 0: result = std::get<0>(data_).get(); break;
          case 1: result = std::get<1>(data_).get(); break;
          case 2: result = std::get<2>(data_).get(); break;
          case 3: result = std::get<3>(data_).get(); break;
          case 4: result = std::get<4>(data_).get(); break;
        }
        return result;
      }()}
  {
  }
  mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>& operator=(mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>&& other) noexcept {
    device_type_ = std::move(other.device_type_);
    extents_ = std::move(other.extents_);
    data_ = std::move(other.data_);
    length_ = std::move(other.size());
    memory_type_ = std::move(other.memory_type_);
    cached_ptr = std::move(other.cached_ptr);
    return *this;
  }
  auto extents() const noexcept { return extents_; }
  HOST DEVICE auto* data_handle() const noexcept { 
      return cached_ptr;
  }

  auto mem_type() const noexcept
  {
    return memory_type_;
  }

  ~mdbuffer() = default;

  HOST DEVICE auto view() const noexcept { 
    if (data_.index() == 0)
      return std::get<0>(data_).view();
    if (data_.index() == 1)
      return std::get<1>(data_).view();
    if (data_.index() == 2)
      return std::get<2>(data_).view();
    if (data_.index() == 3)
      return std::get<3>(data_).view();
    if (data_.index() == 4)
      return std::get<4>(data_).view();
    }

  auto size() {return length_;}
 private:
 auto dev_type() const noexcept
  {
    return device_type_;
  }

  enum device_type device_type_;
  Extents extents_;
  data_store data_;
  size_t length_;
  enum memory_type memory_type_;
  ElementType* cached_ptr;
};

template <bool bounds_check, typename T, typename U, typename DstExtents, typename SrcExtents, typename DstLayoutPolicy, typename SrcLayoutPolicy, template<typename> typename DstContainerPolicy, template<typename> typename SrcContainerPolicy>
detail::const_agnostic_same_t<T, U> copy(raft::resources const& handle,
                                         mdbuffer<T, DstExtents, DstLayoutPolicy, DstContainerPolicy> & dst,
                                         mdbuffer<U, SrcExtents, SrcLayoutPolicy, SrcContainerPolicy> const& src,
                                         size_t dst_offset,
                                         size_t src_offset,
                                         size_t size)
{
  if constexpr (bounds_check) {
    if (src.size() - src_offset < size || dst.size() - dst_offset < size) {
      throw out_of_bounds("Attempted copy to or from mdbuffer of inadequate size");
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

template <bool bounds_check, typename T, typename U, typename DstExtents, typename SrcExtents, typename DstLayoutPolicy, typename SrcLayoutPolicy, template<typename> typename DstContainerPolicy, template<typename> typename SrcContainerPolicy>
detail::const_agnostic_same_t<T, U> copy(raft::resources const& handle,
                                         mdbuffer<T, DstExtents, DstLayoutPolicy, DstContainerPolicy>& dst,
                                         mdbuffer<U, SrcExtents, SrcLayoutPolicy, SrcContainerPolicy> const& src)
{
  copy<bounds_check>(handle, dst, src, 0, 0, src.size());
}
}  // namespace raft