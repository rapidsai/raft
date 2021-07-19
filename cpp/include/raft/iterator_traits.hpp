/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <iterator>
#include <type_traits>

#include <thrust/iterator/iterator_traits.h>

namespace raft {

/**
 * @brief SFINAE check to allow for RAFT APIs to accept any random access buffer. 
 * specializing for T * const and T const * const by using `std::remove_const`
 * these types are not considered random access iterators, but it should
 * be okay for us to do so as we only need random access buffers
 * 
 * Example:
 * template <typename Buf, std::enable_if_t<is_random_access_buffer<Buf>::value> * = nullptr>
 * void raft_prim(const raft::handle& handle, Buf input_begin, Buf input_end,
 *                Buf output_begin) {...}
 */
template <typename T>
struct is_random_access_buffer
  : std::bool_constant<
      std::is_convertible_v<typename thrust::iterator_traits<
                              std::remove_const_t<T>>::iterator_category,
                            std::random_access_iterator_tag>> {};

}  // namespace raft