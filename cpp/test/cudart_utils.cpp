/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <iostream>

namespace raft {

TEST(Raft, Utils)
{
  ASSERT_NO_THROW(ASSERT(1 == 1, "Should not assert!"));
  ASSERT_THROW(ASSERT(1 != 1, "Should assert!"), exception);
  ASSERT_THROW(THROW("Should throw!"), exception);
  ASSERT_NO_THROW(RAFT_CUDA_TRY(cudaFree(nullptr)));
}

}  // namespace raft
