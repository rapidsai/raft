/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <limits.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <raft/sparse/distance/common.h>
#include <raft/sparse/linalg/transpose.h>
#include <raft/sparse/utils.h>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/convert/dense.cuh>
#include <raft/sparse/distance/detail/coo_spmv.cuh>
#include <raft/sparse/distance/detail/operators.cuh>
#include <rmm/device_uvector.hpp>

#include <nvfunctional>

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

template <typename value_idx, typename value_t>
class ip_distances_t : public distances_t<value_t> {
 public:
  /**
         * Computes simple sparse inner product distances as sum(x_y * y_k)
         * @param[in] config specifies inputs, outputs, and sizes
         */
  ip_distances_t(const distances_config_t<value_idx, value_t> &config)
    : config_(&config), coo_rows_b(config.b_nnz, config.handle.get_stream()) {
    raft::sparse::convert::csr_to_coo(config_->b_indptr, config_->b_nrows,
                                      coo_rows_b.data(), config_->b_nnz,
                                      config_->handle.get_stream());
  }

  /**
         * Performs pairwise distance computation and computes output distances
         * @param out_distances dense output matrix (size a_nrows * b_nrows)
         */
  void compute(value_t *out_distances) {
    /**
               * Compute pairwise distances and return dense matrix in row-major format
               */
    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
      out_distances, *config_, coo_rows_b.data(), Product(), Sum(),
      AtomicAdd());
  }

  value_idx *b_rows_coo() { return coo_rows_b.data(); }

  value_t *b_data_coo() { return config_->b_data; }

 private:
  const distances_config_t<value_idx, value_t> *config_;
  rmm::device_uvector<value_idx> coo_rows_b;
};

};  // END namespace detail
};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
