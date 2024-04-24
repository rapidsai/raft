/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_resources.hpp>
#include <raft/matrix/select_k_types.hpp>
#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT

#include <cuda_fp16.h>  // __half

#include <cstdint>  // uint32_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::matrix::detail {

template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              const T* in_val,
              const IdxT* in_idx,
              size_t batch_size,
              size_t len,
              int k,
              T* out_val,
              IdxT* out_idx,
              bool select_min,
              bool sorted       = false,
              SelectAlgo algo   = SelectAlgo::kAuto,
              const IdxT* len_i = nullptr) RAFT_EXPLICIT;
}  // namespace raft::matrix::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_matrix_detail_select_k(T, IdxT)                             \
  extern template void raft::matrix::detail::select_k(raft::resources const& handle, \
                                                      const T* in_val,               \
                                                      const IdxT* in_idx,            \
                                                      size_t batch_size,             \
                                                      size_t len,                    \
                                                      int k,                         \
                                                      T* out_val,                    \
                                                      IdxT* out_idx,                 \
                                                      bool select_min,               \
                                                      bool sorted,                   \
                                                      raft::matrix::SelectAlgo algo, \
                                                      const IdxT* len_i)
instantiate_raft_matrix_detail_select_k(__half, uint32_t);
instantiate_raft_matrix_detail_select_k(__half, int64_t);
instantiate_raft_matrix_detail_select_k(float, int64_t);
instantiate_raft_matrix_detail_select_k(float, uint32_t);
// needed for brute force knn
instantiate_raft_matrix_detail_select_k(float, int);
// We did not have these two for double before, but there are tests for them. We
// therefore include them here.
instantiate_raft_matrix_detail_select_k(double, int64_t);
instantiate_raft_matrix_detail_select_k(double, uint32_t);

#undef instantiate_raft_matrix_detail_select_k
