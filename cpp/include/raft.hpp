/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

/**
 * This file is deprecated and will be removed in release 22.06.
 */
#include "raft/core/handle.hpp"
#include "raft/mdarray.hpp"
#include "raft/span.hpp"

#include <string>

namespace raft {

/* Function for testing RAFT include
 *
 * @return message indicating RAFT has been included succesfully*/
inline std::string test_raft()
{
  std::string status = "RAFT Setup succesfully";
  return status;
}

}  // namespace raft
