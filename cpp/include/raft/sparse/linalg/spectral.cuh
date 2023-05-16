/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#ifndef __SPARSE_SPECTRAL_H
#define __SPARSE_SPECTRAL_H

#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/detail/spectral.cuh>

namespace raft {
namespace sparse {
namespace spectral {

template <typename T>
void fit_embedding(raft::resources const& handle,
                   int* rows,
                   int* cols,
                   T* vals,
                   int nnz,
                   int n,
                   int n_components,
                   T* out,
                   unsigned long long seed = 1234567)
{
  detail::fit_embedding(handle, rows, cols, vals, nnz, n, n_components, out, seed);
}
};  // namespace spectral
};  // namespace sparse
};  // namespace raft

#endif