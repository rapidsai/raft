/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/util/cache_util.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>

namespace raft::matrix::detail {

template <typename m_t, typename idx_t = int>
void printHost(
  const m_t* in, idx_t n_rows, idx_t n_cols, char h_separator = ' ', char v_separator = '\n', )
{
  for (idx_t i = 0; i < n_rows; i++) {
    for (idx_t j = 0; j < n_cols; j++) {
      printf("%1.4f%c", in[j * n_rows + i], j < n_cols - 1 ? h_separator : v_separator);
    }
  }
}

}  // end namespace raft::matrix::detail
