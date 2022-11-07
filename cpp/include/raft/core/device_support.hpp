/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/core/error.hpp>

namespace raft {
#ifdef RAFT_DISABLE_CUDA
auto constexpr static const CUDA_ENABLED = false;
#else
auto constexpr static const CUDA_ENABLED = true;
#endif

struct cuda_unsupported : raft::exception {
  cuda_unsupported() : cuda_unsupported{"CUDA functionality invoked in non-CUDA build"} {}
};

}  // namespace raft
