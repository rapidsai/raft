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

#include <gtest/gtest.h>

#include "../ann_cagra.cuh"

// #if defined RAFT_DISTANCE_COMPILED
// #include <raft/neighbors/specializations.cuh>
// #endif

namespace raft::neighbors::experimental::cagra {

typedef AnnCagraTest<float, float, std::uint32_t> AnnCagraTestF;
TEST_P(AnnCagraTestF, AnnCagra) { this->testCagra(); }

INSTANTIATE_TEST_CASE_P(AnnCagraTest, AnnCagraTestF, ::testing::ValuesIn(inputs));

}  // namespace raft::neighbors::experimental::cagra
