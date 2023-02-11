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

#include "refine.cuh"
#include <common/benchmark.hpp>

#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#include <raft/neighbors/specializations/refine.cuh>
#endif

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.cuh>
#endif

using namespace raft::neighbors;

namespace raft::bench::neighbors {
using refine_float_int64 = RefineAnn<float, float, uint64_t>;
RAFT_BENCH_REGISTER(refine_float_int64, "", getInputs<uint64_t>());
}  // namespace raft::bench::neighbors
