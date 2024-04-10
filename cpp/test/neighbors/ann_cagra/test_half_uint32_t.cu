/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../ann_cagra.cuh"

#include <gtest/gtest.h>

namespace raft::neighbors::cagra {

typedef AnnCagraTest<float, half, std::uint32_t> AnnCagraTestH_U32;
TEST_P(AnnCagraTestH_U32, AnnCagra) { this->testCagra(); }

typedef AnnCagraSortTest<float, half, std::uint32_t> AnnCagraSortTestH_U32;
TEST_P(AnnCagraSortTestH_U32, AnnCagraSort) { this->testCagraSort(); }

typedef AnnCagraFilterTest<float, half, std::uint32_t> AnnCagraFilterTestH_U32;
TEST_P(AnnCagraFilterTestH_U32, AnnCagraFilter)
{
  this->testCagraFilter();
  this->testCagraRemoved();
}

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestH_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraSortTest, AnnCagraSortTestH_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest, AnnCagraFilterTestH_U32,
::testing::ValuesIn(inputs));

}  // namespace raft::neighbors::cagra
