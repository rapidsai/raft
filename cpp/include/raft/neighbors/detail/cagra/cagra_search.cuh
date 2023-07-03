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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/detail/ivf_pq_search.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include "factory.cuh"
#include "search_plan.cuh"
#include "search_single_cta.cuh"

namespace raft::neighbors::experimental::cagra::detail {

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [build](#build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */

template <typename T, typename internal_IdxT, typename IdxT = uint32_t, typename DistanceT = float>
void search_main(raft::resources const& res,
                 search_params params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, internal_IdxT, row_major> queries,
                 raft::device_matrix_view<internal_IdxT, internal_IdxT, row_major> neighbors,
                 raft::device_matrix_view<DistanceT, internal_IdxT, row_major> distances)
{
  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(index.dataset().extent(0)),
                 static_cast<size_t>(index.dataset().extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  RAFT_EXPECTS(queries.extent(1) == index.dim(), "Querise and index dim must match");
  const uint32_t topk = neighbors.extent(1);

  if (params.max_queries == 0) { params.max_queries = queries.extent(0); }

  std::unique_ptr<search_plan_impl<T, internal_IdxT, DistanceT>> plan =
    factory<T, internal_IdxT, DistanceT>::create(
      res, params, index.dim(), index.graph_degree(), topk);

  plan->check(neighbors.extent(1));

  RAFT_LOG_DEBUG("Cagra search");
  const uint32_t max_queries = plan->max_queries;
  const uint32_t query_dim   = queries.extent(1);

  for (unsigned qid = 0; qid < queries.extent(0); qid += max_queries) {
    const uint32_t n_queries = std::min<std::size_t>(max_queries, queries.extent(0) - qid);
    internal_IdxT* _topk_indices_ptr =
      reinterpret_cast<internal_IdxT*>(neighbors.data_handle()) + (topk * qid);
    DistanceT* _topk_distances_ptr = distances.data_handle() + (topk * qid);
    // todo(tfeher): one could keep distances optional and pass nullptr
    const T* _query_ptr = queries.data_handle() + (query_dim * qid);
    const internal_IdxT* _seed_ptr =
      plan->num_seeds > 0
        ? reinterpret_cast<const internal_IdxT*>(plan->dev_seed.data()) + (plan->num_seeds * qid)
        : nullptr;
    uint32_t* _num_executed_iterations = nullptr;

    auto dataset_internal = make_device_strided_matrix_view<const T, internal_IdxT, row_major>(
      index.dataset().data_handle(),
      index.dataset().extent(0),
      index.dataset().extent(1),
      index.dataset().stride(0));
    auto graph_internal =
      raft::make_device_matrix_view<const internal_IdxT, internal_IdxT, row_major>(
        reinterpret_cast<const internal_IdxT*>(index.graph().data_handle()),
        index.graph().extent(0),
        index.graph().extent(1));

    (*plan)(res,
            dataset_internal,
            graph_internal,
            _topk_indices_ptr,
            _topk_distances_ptr,
            _query_ptr,
            n_queries,
            _seed_ptr,
            _num_executed_iterations,
            topk);
  }

  static_assert(std::is_same_v<DistanceT, float>,
                "only float distances are supported at the moment");
  float* dist_out          = distances.data_handle();
  const DistanceT* dist_in = distances.data_handle();
  // We're converting the data from T to DistanceT during distance computation
  // and divide the values by kDivisor. Here we restore the original scale.
  constexpr float kScale = spatial::knn::detail::utils::config<T>::kDivisor /
                           spatial::knn::detail::utils::config<DistanceT>::kDivisor;
  ivf_pq::detail::postprocess_distances(dist_out,
                                        dist_in,
                                        index.metric(),
                                        distances.extent(0),
                                        distances.extent(1),
                                        kScale,
                                        resource::get_cuda_stream(res));
}
/** @} */  // end group cagra

}  // namespace raft::neighbors::experimental::cagra::detail
