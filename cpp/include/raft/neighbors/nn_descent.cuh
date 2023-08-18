/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "detail/nn_descent.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>

namespace raft::neighbors::nn_descent {

/**
 * @defgroup nn-descent CUDA ANN Graph-based gradient descent nearest neighbor
 * @{
 */

/**
 * @brief Build nn-descent Index with dataset in device memory
 * 
 * @tparam T 
 * @tparam IdxT 
 * @param res raft::resources
 * @param params nn_descent::index_params
 * @param dataset raft::device_matrix_view
 * @return index<IdxT> 
 */
template <typename T, typename IdxT = uint32_t>
index<IdxT> build(raft::resources const& res,
                  index_params const& params,
                     raft::device_matrix_view<const T, int64_t, row_major> dataset) {
    return detail::build<T, IdxT>(res, params, dataset);
}

/**
 * @brief Build nn-descent Index with dataset in host memory
 * 
 * @tparam T 
 * @tparam IdxT 
 * @param res raft::resources
 * @param params nn_descent::index_params
 * @param dataset raft::host_matrix_view
 * @return index<IdxT> 
 */
template <typename T, typename IdxT = uint32_t>
index<IdxT> build(raft::resources const& res,
                 index_params const& params,
                     raft::host_matrix_view<const T, int64_t, row_major> dataset) {
    return detail::build<T, IdxT>(res, params, dataset);
}

/** @} */  // end group cagra

}
