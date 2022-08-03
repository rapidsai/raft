/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/core/mdarray.hpp>

#include <rmm/device_uvector.hpp>

#include <type_traits>
#include <vector>

namespace raft::detail {

template <typename T>
struct serial;

template <typename T, typename... ContextArgs>
auto call_serialize(uint8_t* out, const T& obj, ContextArgs&&... args) -> size_t
{
  return detail::serial<T>::to_bytes(out, obj, std::forward<ContextArgs>(args)...);
}

template <typename T, typename... ContextArgs>
auto call_serialize(const T& obj, ContextArgs&&... args) -> std::vector<uint8_t>
{
  std::vector<uint8_t> v(
    call_serialize<T, ContextArgs...>(nullptr, obj, std::forward<ContextArgs>(args)...));
  call_serialize<T, ContextArgs...>(v.data(), obj, std::forward<ContextArgs>(args)...);
  return v;
}

template <typename T, typename... ContextArgs>
auto call_deserialize(T* p, const uint8_t* in, ContextArgs&&... args) -> size_t
{
  return detail::serial<T>::from_bytes(p, in, std::forward<ContextArgs>(args)...);
}

template <typename T, typename... ContextArgs>
auto call_deserialize(T* p, const std::vector<uint8_t>& in, ContextArgs&&... args) -> size_t
{
  return call_deserialize<T, ContextArgs...>(p, in.data(), std::forward<ContextArgs>(args)...);
}

template <typename T, typename... ContextArgs>
auto call_deserialize(const uint8_t* in, ContextArgs&&... args) -> T
{
  union res {
    T value;
    explicit res(const uint8_t* in, ContextArgs&&... args)
    {
      call_deserialize<T, ContextArgs...>(&value, in, std::forward<ContextArgs>(args)...);
    }
    ~res() { value.~T(); }  // NOLINT
  };
  // using a union to avoid initialization of T and force copy elision.
  return res(in, std::forward<ContextArgs>(args)...).value;
}

template <typename T, typename... ContextArgs>
auto call_deserialize(const std::vector<uint8_t>& in, ContextArgs&&... args) -> T
{
  return call_deserialize<T, ContextArgs...>(in.data(), std::forward<ContextArgs>(args)...);
}

template <typename T>
struct serial {
  // Default implementation for all arithmetic types: just write the value by the pointer.
  template <typename S = T>
  static auto to_bytes(uint8_t* out, const S& obj)
    -> std::enable_if_t<std::is_arithmetic_v<S>, size_t>
  {
    if (out) { memcpy(out, &obj, sizeof(S)); }
    return sizeof(S);
  }

  // SFINAE-style failure
  template <typename S = T, typename... ContextArgs>
  static auto to_bytes(uint8_t* out, const S& obj, ContextArgs&&... args)
    -> std::enable_if_t<!std::is_arithmetic_v<S>, size_t>
  {
    static_assert(!std::is_same_v<T, S>, "Serialization is not implemented for this type.");
    return 0;
  }

  // Default implementation for all arithmetic types: just read the value by the pointer.
  template <typename S = T>
  static auto from_bytes(S* p, const uint8_t* in)
    -> std::enable_if_t<std::is_arithmetic_v<S>, size_t>
  {
    memcpy(p, in, sizeof(S));
    return sizeof(S);
  }

  // SFINAE-style failure
  template <typename S = T, typename... ContextArgs>
  static auto from_bytes(S* p, const uint8_t* in, ContextArgs&&... args)
    -> std::enable_if_t<!std::is_arithmetic_v<S>, size_t>
  {
    static_assert(!std::is_same_v<T, S>, "Deserialization is not implemented for this type.");
    return 0;
  }
};

template <typename IndexType, size_t... ExtentsPack>
struct serial<extents<IndexType, ExtentsPack...>> {
  using obj_t = extents<IndexType, ExtentsPack...>;

  static auto to_bytes(uint8_t* out, const obj_t& obj) -> size_t
  {
    if (out) { *reinterpret_cast<obj_t*>(out) = obj; }
    return sizeof(obj_t);
  }

  static auto from_bytes(obj_t* p, const uint8_t* in) -> size_t
  {
    new (p) obj_t{*reinterpret_cast<const obj_t*>(in)};
    return sizeof(obj_t);
  }
};

template <typename ElementType, typename Extents, typename LayoutPolicy>
struct serial<mdarray<ElementType,
                      Extents,
                      LayoutPolicy,
                      detail::device_accessor<detail::device_uvector_policy<ElementType>>>> {
  using obj_t = mdarray<ElementType,
                        Extents,
                        LayoutPolicy,
                        detail::device_accessor<detail::device_uvector_policy<ElementType>>>;

  static auto to_bytes(uint8_t* out, const obj_t& obj, const handle_t& handle) -> size_t
  {
    auto extents_size = call_serialize<Extents>(out, obj.extents());
    auto total_size   = obj.size() * sizeof(ElementType) + extents_size;
    if (out) {
      out += extents_size;
      raft::copy(
        reinterpret_cast<ElementType*>(out), obj.data_handle(), obj.size(), handle.get_stream());
    }
    return total_size;
  }

  static auto from_bytes(obj_t* p,
                         const uint8_t* in,
                         const handle_t& handle,
                         rmm::mr::device_memory_resource* mr = nullptr) -> size_t
  {
    Extents exts;
    auto extents_size = call_deserialize<Extents>(&exts, in);
    in += extents_size;
    typename obj_t::mapping_type layout{exts};
    typename obj_t::container_policy_type policy{handle.get_stream(), mr};
    new (p) obj_t{layout, policy};
    raft::copy(
      p->data_handle(), reinterpret_cast<const ElementType*>(in), p->size(), handle.get_stream());
    return p->size() * sizeof(ElementType) + extents_size;
  }
};

template <typename T>
struct serial<rmm::device_uvector<T>> {
  static auto to_bytes(uint8_t* out,
                       const rmm::device_uvector<T>& obj,
                       rmm::cuda_stream_view stream) -> size_t
  {
    auto pref_size = call_serialize<size_t>(out, obj.size());
    if (out) {
      out += pref_size;
      raft::copy(reinterpret_cast<T*>(out), obj.data(), obj.size(), stream);
    }
    return obj.size() * sizeof(T) + pref_size;
  }

  static auto from_bytes(rmm::device_uvector<T>* p,
                         const uint8_t* in,
                         rmm::cuda_stream_view stream,
                         rmm::mr::device_memory_resource* mr = nullptr) -> size_t
  {
    size_t n;
    auto pref_size = call_deserialize<size_t>(&n, in);
    in += pref_size;
    if (mr) {
      new (p) rmm::device_uvector<T>{n, stream, mr};
    } else {
      new (p) rmm::device_uvector<T>{n, stream};
    }
    raft::copy(p->data(), reinterpret_cast<const T*>(in), p->size(), stream);
    return p->size() * sizeof(T) + pref_size;
  }
};

}  // namespace raft::detail
