/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

<<<<<<< HEAD
#include <raft/neighbors/specializations/ball_cover.cuh>
#include <raft/neighbors/specializations/fused_l2_knn.cuh>
#include <raft/neighbors/specializations/ivf_flat.cuh>
=======
>>>>>>> upstream/branch-23.04
#include <raft/neighbors/specializations/ivf_pq.cuh>
#include <raft/neighbors/specializations/refine.cuh>

#include <raft/cluster/specializations.cuh>
#include <raft/distance/specializations.cuh>
#include <raft/matrix/specializations.cuh>