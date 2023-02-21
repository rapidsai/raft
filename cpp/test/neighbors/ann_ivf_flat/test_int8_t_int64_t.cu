/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "../ann_ivf_flat.cuh"

#if defined RAFT_DISTANCE_COMPILED && defined RAFT_NN_COMPILED
#include <raft/cluster/specializations.cuh>
#include <raft/neighbors/specializations.cuh>
#endif

namespace raft::neighbors::ivf_flat {

typedef AnnIVFFlatTest<float, int8_t, std::int64_t> AnnIVFFlatTestF_int8;
TEST_P(AnnIVFFlatTestF_int8, AnnIVFFlat) { this->testIVFFlat(); }

INSTANTIATE_TEST_CASE_P(AnnIVFFlatTest, AnnIVFFlatTestF_int8, ::testing::ValuesIn(inputs));

}  // namespace raft::neighbors::ivf_flat
