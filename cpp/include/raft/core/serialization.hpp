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
#include <raft/detail/serialization.hpp>

#include <vector>

namespace raft {

/**
 * @brief Write a serializable state of an object to memory.
 *
 * @tparam Type of the serializable object
 *
 * @param handle provides a GPU context for objects, whos state depends on it
 * @param obj the object to be serialized
 * @param[out] out a host pointer to the location where to store the state;
 *   when `nullptr`, the actual data is not written (only the size-to-written is calculated).
 * @return the number of bytes (to be) written by the pointer
 */
template <typename T>
auto serialize(const handle_t& handle, const T& obj, void* out) -> size_t
{
  return detail::call_serialize<T>(handle, obj, reinterpret_cast<uint8_t*>(out));
}

/**
 * @brief Write a serializable state of an object to a host vector.
 *
 * @tparam Type of the serializable object
 *
 * @param handle provides a GPU context for objects, whos state depends on it
 * @param obj the object to be serialized
 * @return the serialized state
 */
template <typename T>
auto serialize(const handle_t& handle, const T& obj) -> std::vector<uint8_t>
{
  return detail::call_serialize<T>(handle, obj);
}

/**
 * @brief Read a serializable state of an object from memory.
 *
 * @tparam Type of the serializable object
 *
 * @param handle provides a GPU context for objects, whos state depends on it.
 * @param[out] p an unitialized host pointer to a location where the object should be created.
 * @param[in] in a host pointer to the location where the state should be read from.
 */
template <typename T>
void deserialize(const handle_t& handle, T* p, const void* in)
{
  return detail::call_deserialize<T>(handle, p, reinterpret_cast<const uint8_t*>(in));
}

/**
 * @brief Read a serializable state of an object from a vector.
 *
 * @tparam Type of the serializable object
 *
 * @param handle provides a GPU context for objects, whos state depends on it.
 * @param[out] p an unitialized host pointer to a location where the object should be created.
 * @param in a vector where the state should be read from.
 */
template <typename T>
void deserialize(const handle_t& handle, T* p, const std::vector<uint8_t>& in)
{
  return detail::call_deserialize<T>(handle, p, in);
}

/**
 * @brief Read a serializable state of an object.
 *
 * @tparam Type of the serializable object
 *
 * @param handle provides a GPU context for objects, whos state depends on it.
 * @param[in] in a host pointer to the location where the state should be read from.
 * @return the deserialized object;
 */
template <typename T>
auto deserialize(const handle_t& handle, const void* in) -> T
{
  return detail::call_deserialize<T>(handle, reinterpret_cast<const uint8_t*>(in));
}

/**
 * @brief Read a serializable state of an object.
 *
 * @tparam Type of the serializable object
 *
 * @param handle provides a GPU context for objects, whos state depends on it.
 * @param in a vector where the state should be read from.
 * @return the deserialized object;
 */
template <typename T>
auto deserialize(const handle_t& handle, const std::vector<uint8_t>& in) -> T
{
  return detail::call_deserialize<T>(handle, in);
}

}  // namespace raft
