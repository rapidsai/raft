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

#ifndef __KNN_SPECIALIZATIONS_H
#define __KNN_SPECIALIZATIONS_H

#pragma once

#include <raft/neighbors/specializations/ball_cover.cuh>
#include <raft/neighbors/specializations/fused_l2_knn.cuh>
#include <raft/neighbors/specializations/ivf_pq_build.cuh>
#include <raft/neighbors/specializations/knn.cuh>

#include <raft/neighbors/specializations/detail/ivf_pq_search.cuh>

#endif
