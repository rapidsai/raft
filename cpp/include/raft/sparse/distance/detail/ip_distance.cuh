/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "common.hpp"

#include <raft/core/operators.cuh>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/distance/detail/coo_spmv.cuh>
#include <raft/sparse/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <limits.h>

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
  ip_distances_t(const distances_config_t<value_idx, value_t>& config)
    : config_(&config), coo_rows_b(config.b_nnz, resource::get_cuda_stream(config.handle))
  {
    raft::sparse::convert::csr_to_coo(config_->b_indptr,
                                      config_->b_nrows,
                                      coo_rows_b.data(),
                                      config_->b_nnz,
                                      resource::get_cuda_stream(config_->handle));
  }

  /**
   * Performs pairwise distance computation and computes output distances
   * @param out_distances dense output matrix (size a_nrows * b_nrows)
   */
  void compute(value_t* out_distances)
  {
    /**
     * Compute pairwise distances and return dense matrix in row-major format
     */
    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(out_distances,
                                                               *config_,
                                                               coo_rows_b.data(),
                                                               raft::mul_op(),
                                                               raft::add_op(),
                                                               raft::atomic_add_op());
  }

  value_idx* b_rows_coo() { return coo_rows_b.data(); }

  value_t* b_data_coo() { return config_->b_data; }

 private:
  const distances_config_t<value_idx, value_t>* config_;
  rmm::device_uvector<value_idx> coo_rows_b;
};

};  // END namespace detail
};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
