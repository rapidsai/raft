/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __SPARSE_SPECTRAL_H
#define __SPARSE_SPECTRAL_H

#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/detail/spectral.cuh>

namespace raft {
namespace sparse {
namespace spectral {

template <typename T, typename IndT, typename nnz_t>
void fit_embedding(raft::resources const& handle,
                   int* rows,
                   int* cols,
                   T* vals,
                   nnz_t nnz,
                   IndT n,
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
