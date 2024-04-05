/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/sparse/matrix/detail/select_k-inl.cuh>

#define instantiate_raft_sparse_matrix_detail_select_k(T, IdxT)       \
  template void raft::sparse::matrix::detail::select_k(               \
    raft::resources const& handle,                                    \
    raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> in_val,   \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> in_idx, \
    raft::device_matrix_view<T, IdxT, raft::row_major> out_val,       \
    raft::device_matrix_view<IdxT, IdxT, raft::row_major> out_idx,    \
    bool select_min,                                                  \
    bool sorted,                                                      \
    raft::matrix::SelectAlgo algo)

instantiate_raft_sparse_matrix_detail_select_k(__half, int64_t);

#undef instantiate_raft_sparse_matrix_detail_select_k
