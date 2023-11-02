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
#include "raft_cagra_hnswlib_wrapper.h"

namespace raft::bench::ann {
template class RaftCagraHnswlib<uint8_t, uint32_t>;
template class RaftCagraHnswlib<int8_t, uint32_t>;
template class RaftCagraHnswlib<float, uint32_t>;
}  // namespace raft::bench::ann
