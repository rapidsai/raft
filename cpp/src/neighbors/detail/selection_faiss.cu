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

#include <cstddef>  // size_t
#include <cstdint>  // uint32_t
#include <raft/neighbors/detail/selection_faiss-inl.cuh>

#define instantiate_raft_neighbors_detail_select_k(payload_t, key_t)    \
  template void raft::neighbors::detail::select_k(const key_t* inK,     \
                                                  const payload_t* inV, \
                                                  size_t n_rows,        \
                                                  size_t n_cols,        \
                                                  key_t* outK,          \
                                                  payload_t* outV,      \
                                                  bool select_min,      \
                                                  int k,                \
                                                  cudaStream_t stream)

// @benfred: Not sure if this is correct. Should I not flip float and uint32_t?
// It seems weird that float is the key and uint32_t is the payload type.
instantiate_raft_neighbors_detail_select_k(uint32_t, float);
instantiate_raft_neighbors_detail_select_k(int32_t, float);
instantiate_raft_neighbors_detail_select_k(long, float);
// Needed by the tests
instantiate_raft_neighbors_detail_select_k(size_t, double);
// test/neighbors/selection.cu
instantiate_raft_neighbors_detail_select_k(int, double);
instantiate_raft_neighbors_detail_select_k(size_t, float);

#undef instantiate_raft_neighbors_detail_select_k
