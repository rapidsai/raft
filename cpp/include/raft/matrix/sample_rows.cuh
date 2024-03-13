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

#include <raft/core/logger.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft::matrix {

/** Select rows randomly from input and copy to output. */
template <typename T, typename IdxT = int64_t>
void sample_rows(raft::resources const& res,
                 const T* input,
                 IdxT n_rows_input,
                 raft::device_matrix_view<T, IdxT> output,
                 RngState random_state)
{
  detail::sample_rows(res, input, n_rows_input, output, random_state);
}

/** Subsample the dataset to create a training set*/
template <typename T, typename IdxT = int64_t>
raft::device_matrix<T, IdxT> sample_rows(raft::resources const& res,
                                         const T* input,
                                         IdxT n_rows_input,
                                         IdxT n_train,
                                         IdxT n_dim,
                                         RngState random_state)
{
  auto output = raft::make_device_matrix<T, IdxT>(res, n_train, n_dim);
  detail::sample_rows(res, input, n_rows_input, output, random_state);
  return output;
}
}  // namespace raft::matrix
