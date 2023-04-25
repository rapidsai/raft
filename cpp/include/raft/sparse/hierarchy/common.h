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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#pragma once

#pragma message(__FILE__                                                    \
                  " is deprecated and will be removed in a future release." \
                  " Please use raft/cluster/single_linkage_types.hpp instead.")

#include <raft/cluster/single_linkage_types.hpp>

namespace raft::hierarchy {
using raft::cluster::linkage_output;
using raft::cluster::linkage_output_int;
using raft::cluster::linkage_output_int64;
using raft::cluster::LinkageDistance;
}  // namespace raft::hierarchy