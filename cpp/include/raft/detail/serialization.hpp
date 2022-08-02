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

#include <type_traits>
#include <vector>

namespace raft::detail {

template <typename T>
struct serialize;

template <typename T>
struct deserialize;

template <typename T>
auto call_serialize(const handle_t& handle, const T& obj, uint8_t* out) -> size_t
{
  return serialize<T>::run(handle, obj, out);
}

template <typename T>
auto call_serialize(const handle_t& handle, const T& obj) -> std::vector<uint8_t>
{
  std::vector<uint8_t> v(call_serialize(handle, obj, nullptr));
  call_serialize<T>(handle, obj, v.data());
  return v;
}

template <typename T>
void call_deserialize(const handle_t& handle, T* p, const uint8_t* in)
{
  return detail::deserialize<T>::run(handle, p, in);
}

template <typename T>
void call_deserialize(const handle_t& handle, T* p, const std::vector<uint8_t>& in)
{
  return call_deserialize<T>(handle, p, in.data());
}

template <typename T>
auto call_deserialize(const handle_t& handle, const uint8_t* in) -> T
{
  union res {
    uint8_t bytes[sizeof(T)];  // NOLINT
    T value;
    res(const handle_t& handle, const uint8_t* in) { call_deserialize(handle, &value, in); }
    ~res() { value.~T(); }  // NOLINT
  };
  // using a union to avoid initialization of T and force copy elision.
  return res(handle, in).value;
}

template <typename T>
auto call_deserialize(const handle_t& handle, const std::vector<uint8_t>& in) -> T
{
  return call_deserialize<T>(handle, in.data());
}

template <typename T>
struct serialize {
  // Default implementation for all arithmetic types: just write the value by the pointer.
  template <typename S = T>
  static auto run(const handle_t& handle, const S& obj, uint8_t* out)
    -> std::enable_if_t<std::is_arithmetic_v<S>, size_t>
  {
    if (out) { *reinterpret_cast<T*>(out) = obj; }
    return sizeof(obj);
  }

  // SFINAE-style failure
  template <typename S = T>
  static auto run(const handle_t& handle, const S& obj, uint8_t* out)
    -> std::enable_if_t<!std::is_arithmetic_v<S>, size_t>
  {
    static_assert(!std::is_same_v<T, S>, "Serialization is not implemented for this type.");
    return 0;
  }
};

template <typename T>
struct deserialize {
  // Default implementation for all arithmetic types: just read the value by the pointer.
  template <typename S = T>
  static auto run(const handle_t& handle, T* p, const uint8_t* in)
    -> std::enable_if_t<std::is_arithmetic_v<S>>
  {
    *p = *reinterpret_cast<const T*>(in);
  }

  // SFINAE-style failure
  template <typename S = T>
  static auto run(const handle_t& handle, T* p, const uint8_t* in)
    -> std::enable_if_t<!std::is_arithmetic_v<S>>
  {
    static_assert(!std::is_same_v<T, S>, "Deserialization is not implemented for this type.");
  }
};

template <typename IndexType, size_t... ExtentsPack>
struct serialize<extents<IndexType, ExtentsPack...>> {
  using obj_t = extents<IndexType, ExtentsPack...>;
  static auto run(const handle_t& handle, const obj_t& obj, uint8_t* out) -> size_t
  {
    if (out) { *reinterpret_cast<obj_t*>(out) = obj; }
    return sizeof(obj_t);
  }
};

template <typename IndexType, size_t... ExtentsPack>
struct deserialize<extents<IndexType, ExtentsPack...>> {
  using obj_t = extents<IndexType, ExtentsPack...>;
  static void run(const handle_t& handle, obj_t* p, const uint8_t* in)
  {
    new (p) obj_t{*reinterpret_cast<const obj_t*>(in)};
  }
};

template <typename ElementType, typename Extents, typename LayoutPolicy>
struct serialize<mdarray<ElementType,
                         Extents,
                         LayoutPolicy,
                         detail::device_accessor<detail::device_uvector_policy<ElementType>>>> {
  using obj_t = mdarray<ElementType,
                        Extents,
                        LayoutPolicy,
                        detail::device_accessor<detail::device_uvector_policy<ElementType>>>;
  static auto run(const handle_t& handle, const obj_t& obj, uint8_t* out) -> size_t
  {
    auto extents_size = call_serialize(handle, obj.extents(), out);
    auto total_size   = obj.size() * sizeof(ElementType);
    if (out) {
      out += extents_size;
      raft::copy(
        reinterpret_cast<ElementType*>(out), obj.data_handle(), obj.size(), handle.get_stream());
    }
    return total_size;
  }
};

template <typename ElementType, typename Extents, typename LayoutPolicy>
struct deserialize<mdarray<ElementType,
                           Extents,
                           LayoutPolicy,
                           detail::device_accessor<detail::device_uvector_policy<ElementType>>>> {
  using obj_t = mdarray<ElementType,
                        Extents,
                        LayoutPolicy,
                        detail::device_accessor<detail::device_uvector_policy<ElementType>>>;
  static void run(const handle_t& handle, obj_t* p, const uint8_t* in)
  {
    auto exts         = call_deserialize<Extents>(handle, in);
    auto extents_size = call_serialize(handle, exts, nullptr);
    in += extents_size;
    typename obj_t::mapping_type layout{exts};
    typename obj_t::container_policy_type policy{handle.get_stream()};
    new (p) obj_t{layout, policy};
    raft::copy(
      p->data_handle(), reinterpret_cast<const ElementType*>(in), p->size(), handle.get_stream());
  }
};

}  // namespace raft::detail
