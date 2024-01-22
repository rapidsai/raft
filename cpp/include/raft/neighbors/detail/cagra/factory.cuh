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

#include "search_multi_cta.cuh"
#include "search_multi_kernel.cuh"
#include "search_plan.cuh"
#include "search_single_cta.cuh"
#include <raft/neighbors/sample_filter_types.hpp>

namespace raft::neighbors::cagra::detail {

template <typename T,
          typename IdxT               = uint32_t,
          typename DistanceT          = float,
          typename CagraSampleFilterT = raft::neighbors::filtering::none_cagra_sample_filter>
class factory {
 public:
  /**
   * Create a search structure for dataset with dim features.
   */
  static std::unique_ptr<search_plan_impl<T, IdxT, DistanceT, CagraSampleFilterT>> create(
    raft::resources const& res,
    search_params const& params,
    int64_t dim,
    int64_t graph_degree,
    uint32_t topk)
  {
    search_plan_impl_base plan(params, dim, graph_degree, topk);
    switch (plan.dataset_block_dim) {
      case 128:
        switch (plan.team_size) {
          case 8: return dispatch_kernel<128, 8>(res, plan); break;
          default: THROW("Incorrect team size %lu", plan.team_size);
        }
        break;
      case 256:
        switch (plan.team_size) {
          case 16: return dispatch_kernel<256, 16>(res, plan); break;
          default: THROW("Incorrect team size %lu", plan.team_size);
        }
        break;
      case 512:
        switch (plan.team_size) {
          case 32: return dispatch_kernel<512, 32>(res, plan); break;
          default: THROW("Incorrect team size %lu", plan.team_size);
        }
        break;
      default: THROW("Incorrect dataset_block_dim (%lu)\n", plan.dataset_block_dim);
    }
    return std::unique_ptr<search_plan_impl<T, IdxT, DistanceT, CagraSampleFilterT>>();
  }

 private:
  template <unsigned DATASET_BLOCK_DIM, unsigned TEAM_SIZE>
  static std::unique_ptr<search_plan_impl<T, IdxT, DistanceT, CagraSampleFilterT>> dispatch_kernel(
    raft::resources const& res, search_plan_impl_base& plan)
  {
    if (plan.algo == search_algo::SINGLE_CTA) {
      return std::unique_ptr<search_plan_impl<T, IdxT, DistanceT, CagraSampleFilterT>>(
        new single_cta_search::
          search<TEAM_SIZE, DATASET_BLOCK_DIM, T, IdxT, DistanceT, CagraSampleFilterT>(
            res, plan, plan.dim, plan.graph_degree, plan.topk));
    } else if (plan.algo == search_algo::MULTI_CTA) {
      return std::unique_ptr<search_plan_impl<T, IdxT, DistanceT, CagraSampleFilterT>>(
        new multi_cta_search::
          search<TEAM_SIZE, DATASET_BLOCK_DIM, T, IdxT, DistanceT, CagraSampleFilterT>(
            res, plan, plan.dim, plan.graph_degree, plan.topk));
    } else {
      return std::unique_ptr<search_plan_impl<T, IdxT, DistanceT, CagraSampleFilterT>>(
        new multi_kernel_search::
          search<TEAM_SIZE, DATASET_BLOCK_DIM, T, IdxT, DistanceT, CagraSampleFilterT>(
            res, plan, plan.dim, plan.graph_degree, plan.topk));
    }
  }
};
};  // namespace raft::neighbors::cagra::detail
