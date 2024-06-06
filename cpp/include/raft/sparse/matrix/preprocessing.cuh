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

template <typename T1, typename T2, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_vector_view<T1, IdxT> rows,
                 raft::device_vector_view<T1, IdxT> columns,
                 raft::device_vector_view<T2, IdxT> values,
                 raft::device_vector_view<T2, IdxT> values_out,
                 float k_param = 1.6f,
                 float b_param = 0.75)
{
  return matrix::detail::encode_bm25<T1, T2, IdxT>(
    handle, rows, columns, values, values_out, k_param, b_param);
}
}  // namespace raft::sparse::matrix
