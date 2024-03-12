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

typedef AnnCagraTest<float, std::uint8_t, std::uint32_t> AnnCagraTestU8_U32;
TEST_P(AnnCagraTestU8_U32, AnnCagra) { this->testCagra(); }

typedef AnnCagraSortTest<float, std::uint8_t, std::uint32_t> AnnCagraSortTestU8_U32;
TEST_P(AnnCagraSortTestU8_U32, AnnCagraSort) { this->testCagraSort(); }

typedef AnnCagraFilterTest<float, std::uint8_t, std::uint32_t> AnnCagraFilterTestU8_U32;
TEST_P(AnnCagraFilterTestU8_U32, AnnCagraSort)
{
  this->testCagraFilter();
  this->testCagraRemoved();
}

typedef AnnCagraAddNodesTest<float, std::uint8_t, std::uint32_t> AnnCagraAddNodesTestU8_U32;
TEST_P(AnnCagraAddNodesTestU8_U32, AnnCagra) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestU8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraSortTest, AnnCagraSortTestU8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraFilterTest, AnnCagraFilterTestU8_U32, ::testing::ValuesIn(inputs));
INSTANTIATE_TEST_CASE_P(AnnCagraAddNodesTest,
                        AnnCagraAddNodesTestU8_U32,
                        ::testing::ValuesIn(inputs));

}  // namespace raft::neighbors::cagra
