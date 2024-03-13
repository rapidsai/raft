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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/sample_rows.cuh>
#include <raft/random/rng.cuh>

namespace raft::matrix {

/** Select rows randomly from input and copy to output. */
template <typename T, typename IdxT = int64_t, typename accessor>
void sample_rows(raft::resources const& res,
                 random::RngState random_state,
                 mdspan<const T, matrix_extent<IdxT>, row_major, accessor> dataset,
                 raft::device_matrix_view<T, IdxT> output)
{
  detail::sample_rows<T, IdxT>(res, random_state, dataset.data_handle(), dataset.extent(0), output);
}

/** Subsample the dataset to create a training set*/
template <typename T, typename IdxT = int64_t, typename accessor>
raft::device_matrix<T, IdxT> sample_rows(
  raft::resources const& res,
  random::RngState random_state,
  mdspan<const T, matrix_extent<IdxT>, row_major, accessor> dataset,
  IdxT n_samples)
{
  auto output = raft::make_device_matrix<T, IdxT>(res, n_samples, dataset.extent(1));
  sample_rows(res, random_state, dataset, output.view());
  return output;
}

}  // namespace raft::matrix
