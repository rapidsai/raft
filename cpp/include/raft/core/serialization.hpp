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
 * @tparam T type of the serializable object
 * @tparam ContextArgs types of context required for serialization
 *
 * @param[out] out a host pointer to the location where to store the state;
 *   when `nullptr`, the actual data is not written (only the size-to-written is calculated).
 * @param obj the object to be serialized
 * @param args context required for serialization
 * @return the number of bytes (to be) written by the pointer
 */
template <typename T, typename... ContextArgs>
auto serialize(uint8_t* out, const T& obj, ContextArgs&&... args) -> size_t
{
  return detail::call_serialize<T, ContextArgs...>(out, obj, std::forward<ContextArgs>(args)...);
}

/**
 * @brief Write a serializable state of an object to a host vector.
 *
 * @tparam T type of the serializable object
 * @tparam ContextArgs types of context required for serialization
 *
 * @param obj the object to be serialized
 * @param args context required for serialization
 * @return the serialized state
 */
template <typename T, typename... ContextArgs>
auto serialize(const T& obj, ContextArgs&&... args) -> std::vector<uint8_t>
{
  return detail::call_serialize<T, ContextArgs...>(obj, std::forward<ContextArgs>(args)...);
}

/**
 * @brief Read a serializable state of an object from memory.
 *
 * @tparam T type of the serializable object
 * @tparam ContextArgs types of context required for serialization
 *
 * @param[out] p an unitialized host pointer to a location where the object should be created.
 * @param[in] in a host pointer to the location where the state should be read from.
 * @param args context required for serialization
 * @return the number of bytes read by the pointer
 */
template <typename T, typename... ContextArgs>
auto deserialize(T* p, const void* in, ContextArgs&&... args) -> size_t
{
  return detail::call_deserialize<T, ContextArgs...>(
    p, reinterpret_cast<const uint8_t*>(in), std::forward<ContextArgs>(args)...);
}

/**
 * @brief Read a serializable state of an object from a vector.
 *
 * @tparam T type of the serializable object
 * @tparam ContextArgs types of context required for serialization
 *
 * @param[out] p an unitialized host pointer to a location where the object should be created.
 * @param in a vector where the state should be read from.
 * @param args context required for serialization
 * @return the number of bytes read from the vector
 */
template <typename T, typename... ContextArgs>
auto deserialize(T* p, const std::vector<uint8_t>& in, ContextArgs&&... args) -> size_t
{
  return detail::call_deserialize<T, ContextArgs...>(p, in, std::forward<ContextArgs>(args)...);
}

/**
 * @brief Read a serializable state of an object.
 *
 * @tparam T type of the serializable object
 * @tparam ContextArgs types of context required for serialization
 *
 * @param[in] in a host pointer to the location where the state should be read from.
 * @param args context required for serialization
 * @return the deserialized object;
 */
template <typename T, typename... ContextArgs>
auto deserialize(const void* in, ContextArgs&&... args) -> T
{
  return detail::call_deserialize<T, ContextArgs...>(reinterpret_cast<const uint8_t*>(in),
                                                     std::forward<ContextArgs>(args)...);
}

/**
 * @brief Read a serializable state of an object.
 *
 * @tparam T type of the serializable object
 * @tparam ContextArgs types of context required for serialization
 *
 * @param in a vector where the state should be read from.
 * @param args context required for serialization
 * @return the deserialized object;
 */
template <typename T, typename... ContextArgs>
auto deserialize(const std::vector<uint8_t>& in, ContextArgs&&... args) -> T
{
  return detail::call_deserialize<T, ContextArgs...>(in, std::forward<ContextArgs>(args)...);
}

}  // namespace raft
