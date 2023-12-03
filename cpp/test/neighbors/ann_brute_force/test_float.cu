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

#include "../ann_brute_force.cuh"

namespace raft::neighbors::brute_force {

using AnnBruteForceTest_float = AnnBruteForceTest<float, float, std::int64_t>;
TEST_P(AnnBruteForceTest_float, AnnBruteForce) { this->testBruteForce(); }

INSTANTIATE_TEST_CASE_P(AnnBruteForceTest, AnnBruteForceTest_float, ::testing::ValuesIn(inputs));

}  // namespace raft::neighbors::brute_force
