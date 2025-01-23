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

#pragma once

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/detail/preprocessing.cuh>

#include <optional>

namespace raft::sparse::matrix {

class SparseEncoder {
 public:
  SparseEncoder(int vocab_size);
  template <typename ValueType = float, typename IndexType = int>
  void fit(raft::resources& handle,
           raft::device_coo_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> coo_in);
  template <typename ValueType = float, typename IndexType = int>
  void fit(raft::resources& handle,
           raft::device_csr_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> csr_in);
  template <typename ValueType = float, typename IndexType = int>
  void transform(raft::resources& handle,
                 raft::device_csr_matrix<ValueType,
                                         IndexType,
                                         IndexType,
                                         IndexType,
                                         raft::device_uvector_policy,
                                         raft::PRESERVING> csr_in,
                 float* results,
                 bool bm25_on,
                 float k_param = 1.6f,
                 float b_param = 0.75f);
  template <typename ValueType = float, typename IndexType = int>
  void transform(raft::resources& handle,
                 raft::device_coo_matrix<ValueType,
                                         IndexType,
                                         IndexType,
                                         IndexType,
                                         raft::device_uvector_policy,
                                         raft::PRESERVING> coo_in,
                 float* results,
                 bool bm25_on,
                 float k_param = 1.6f,
                 float b_param = 0.75f);
};

}  // namespace raft::sparse::matrix
