/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "compute_distance_vpq.cuh"
#include "factory.cuh"
#include "search_plan.cuh"
#include "search_single_cta.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/ivf_common.cuh>
#include <raft/neighbors/detail/ivf_pq_search.cuh>
#include <raft/neighbors/sample_filter_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace raft::neighbors::cagra::detail {

template <class CagraSampleFilterT>
struct CagraSampleFilterWithQueryIdOffset {
  const uint32_t offset;
  CagraSampleFilterT filter;

  CagraSampleFilterWithQueryIdOffset(const uint32_t offset, const CagraSampleFilterT filter)
    : offset(offset), filter(filter)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id + offset, sample_id);
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterT_Selector {
  using type = CagraSampleFilterWithQueryIdOffset<CagraSampleFilterT>;
};
template <>
struct CagraSampleFilterT_Selector<raft::neighbors::filtering::none_cagra_sample_filter> {
  using type = raft::neighbors::filtering::none_cagra_sample_filter;
};

// A helper function to set a query id offset
template <class CagraSampleFilterT>
inline typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type set_offset(
  CagraSampleFilterT filter, const uint32_t offset)
{
  typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type new_filter(offset, filter);
  return new_filter;
}
template <>
inline
  typename CagraSampleFilterT_Selector<raft::neighbors::filtering::none_cagra_sample_filter>::type
  set_offset<raft::neighbors::filtering::none_cagra_sample_filter>(
    raft::neighbors::filtering::none_cagra_sample_filter filter, const uint32_t)
{
  return filter;
}

template <typename DatasetDescriptorT, typename CagraSampleFilterT>
void search_main_core(
  raft::resources const& res,
  search_params params,
  DatasetDescriptorT dataset_desc,
  raft::device_matrix_view<const typename DatasetDescriptorT::INDEX_T, int64_t, row_major> graph,
  raft::device_matrix_view<const typename DatasetDescriptorT::DATA_T, int64_t, row_major> queries,
  raft::device_matrix_view<typename DatasetDescriptorT::INDEX_T, int64_t, row_major> neighbors,
  raft::device_matrix_view<typename DatasetDescriptorT::DISTANCE_T, int64_t, row_major> distances,
  CagraSampleFilterT sample_filter    = CagraSampleFilterT(),
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded)
{
  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(dataset_desc.size),
                 static_cast<size_t>(dataset_desc.dim));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  RAFT_EXPECTS(queries.extent(1) == dataset_desc.dim, "Queries and index dim must match");
  const uint32_t topk = neighbors.extent(1);

  cudaDeviceProp deviceProp = resource::get_device_properties(res);
  if (params.max_queries == 0) {
    params.max_queries = std::min<size_t>(queries.extent(0), deviceProp.maxGridSize[1]);
  }

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "cagra::search(max_queries = %u, k = %u, dim = %zu)",
    params.max_queries,
    topk,
    dataset_desc.dim);

  using CagraSampleFilterT_s = typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type;
  std::unique_ptr<search_plan_impl<DatasetDescriptorT, CagraSampleFilterT_s>> plan =
    factory<DatasetDescriptorT, CagraSampleFilterT_s>::create(
      res, params, dataset_desc.dim, graph.extent(1), topk, metric);

  plan->check(topk);

  RAFT_LOG_DEBUG("Cagra search");
  const uint32_t max_queries = plan->max_queries;
  const uint32_t query_dim   = queries.extent(1);

  for (unsigned qid = 0; qid < queries.extent(0); qid += max_queries) {
    const uint32_t n_queries = std::min<std::size_t>(max_queries, queries.extent(0) - qid);
    auto _topk_indices_ptr =
      reinterpret_cast<typename DatasetDescriptorT::INDEX_T*>(neighbors.data_handle()) +
      (topk * qid);
    auto _topk_distances_ptr = distances.data_handle() + (topk * qid);
    // todo(tfeher): one could keep distances optional and pass nullptr
    const auto* _query_ptr = queries.data_handle() + (query_dim * qid);
    const auto* _seed_ptr =
      plan->num_seeds > 0
        ? reinterpret_cast<const typename DatasetDescriptorT::INDEX_T*>(plan->dev_seed.data()) +
            (plan->num_seeds * qid)
        : nullptr;
    uint32_t* _num_executed_iterations = nullptr;

    (*plan)(res,
            dataset_desc,
            graph,
            _topk_indices_ptr,
            _topk_distances_ptr,
            _query_ptr,
            n_queries,
            _seed_ptr,
            _num_executed_iterations,
            topk,
            set_offset(sample_filter, qid));
  }
}

template <class T,
          class DatasetT,
          class DatasetIdxT,
          class InternalIdxT,
          class DistanceT,
          class CagraSampleFilterT>
void launch_vpq_search_main_core(
  raft::resources const& res,
  const vpq_dataset<DatasetT, DatasetIdxT>* vpq_dset,
  search_params params,
  raft::device_matrix_view<const InternalIdxT, int64_t, row_major> graph,
  raft::device_matrix_view<const T, int64_t, row_major> queries,
  raft::device_matrix_view<InternalIdxT, int64_t, row_major> neighbors,
  raft::device_matrix_view<DistanceT, int64_t, row_major> distances,
  CagraSampleFilterT sample_filter,
  const raft::distance::DistanceType metric)
{
  RAFT_EXPECTS(vpq_dset->pq_bits() == 8, "Only pq_bits = 8 is supported for now");
  RAFT_EXPECTS(vpq_dset->pq_len() == 2 || vpq_dset->pq_len() == 4,
               "Only pq_len 2 or 4 is supported for now");
  RAFT_EXPECTS(vpq_dset->dim() % vpq_dset->pq_dim() == 0,
               "dim must be a multiple of pq_dim at the moment");

  const float vq_scale = 1.0f;
  const float pq_scale = 1.0f;

  if (vpq_dset->pq_bits() == 8) {
    if (vpq_dset->pq_len() == 2) {
      using dataset_desc_t = cagra_q_dataset_descriptor_t<T,
                                                          DatasetT,
                                                          8 /*PQ bit*/,
                                                          2 /* Subspace dimension*/,
                                                          DistanceT,
                                                          InternalIdxT>;
      dataset_desc_t dataset_desc(vpq_dset->data.data_handle(),
                                  vpq_dset->encoded_row_length(),
                                  vpq_dset->pq_dim(),
                                  vpq_dset->vq_code_book.data_handle(),
                                  vq_scale,
                                  vpq_dset->pq_code_book.data_handle(),
                                  pq_scale,
                                  size_t(vpq_dset->n_rows()),
                                  vpq_dset->dim());
      search_main_core(
        res, params, dataset_desc, graph, queries, neighbors, distances, sample_filter, metric);
    } else if (vpq_dset->pq_len() == 4) {
      using dataset_desc_t = cagra_q_dataset_descriptor_t<T,
                                                          DatasetT,
                                                          8 /*PQ bit*/,
                                                          4 /* Subspace dimension*/,
                                                          DistanceT,
                                                          InternalIdxT>;
      dataset_desc_t dataset_desc(vpq_dset->data.data_handle(),
                                  vpq_dset->encoded_row_length(),
                                  vpq_dset->pq_dim(),
                                  vpq_dset->vq_code_book.data_handle(),
                                  vq_scale,
                                  vpq_dset->pq_code_book.data_handle(),
                                  pq_scale,
                                  size_t(vpq_dset->n_rows()),
                                  vpq_dset->dim());
      search_main_core(
        res, params, dataset_desc, graph, queries, neighbors, distances, sample_filter, metric);
    } else {
      RAFT_FAIL("Subspace dimension must be 2 or 4");
    }
  } else {
    RAFT_FAIL("Only 8-bit PQ is supported now");
  }
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [build](#build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of database vector indices
 * @tparam internal_IdxT during search we map IdxT to internal_IdxT, this way we do not need
 * separate kernels for int/uint.
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
template <typename T,
          typename InternalIdxT,
          typename CagraSampleFilterT,
          typename IdxT      = uint32_t,
          typename DistanceT = float>
void search_main(raft::resources const& res,
                 search_params params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, int64_t, row_major> queries,
                 raft::device_matrix_view<InternalIdxT, int64_t, row_major> neighbors,
                 raft::device_matrix_view<DistanceT, int64_t, row_major> distances,
                 CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  const auto& graph   = index.graph();
  auto graph_internal = raft::make_device_matrix_view<const InternalIdxT, int64_t, row_major>(
    reinterpret_cast<const InternalIdxT*>(graph.data_handle()), graph.extent(0), graph.extent(1));

  // n_rows has the same type as the dataset index (the array extents type)
  using ds_idx_type = decltype(index.data().n_rows());
  // Dispatch search parameters based on the dataset kind.
  if (auto* strided_dset = dynamic_cast<const strided_dataset<T, ds_idx_type>*>(&index.data());
      strided_dset != nullptr) {
    // Set TEAM_SIZE and DATASET_BLOCK_SIZE to zero tentatively since these parameters cannot be
    // determined here. They are set just before kernel launch.
    using dataset_desc_t = standard_dataset_descriptor_t<T, InternalIdxT, DistanceT>;
    // Search using a plain (strided) row-major dataset
    const dataset_desc_t dataset_desc(strided_dset->view().data_handle(),
                                      strided_dset->n_rows(),
                                      strided_dset->dim(),
                                      strided_dset->stride());
    search_main_core<dataset_desc_t, CagraSampleFilterT>(res,
                                                         params,
                                                         dataset_desc,
                                                         graph_internal,
                                                         queries,
                                                         neighbors,
                                                         distances,
                                                         sample_filter,
                                                         index.metric());
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<float, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    // Search using a compressed dataset
    RAFT_FAIL("FP32 VPQ dataset support is coming soon");
  } else if (auto* vpq_dset = dynamic_cast<const vpq_dataset<half, ds_idx_type>*>(&index.data());
             vpq_dset != nullptr) {
    launch_vpq_search_main_core<T, half, ds_idx_type, InternalIdxT, DistanceT, CagraSampleFilterT>(
      res,
      vpq_dset,
      params,
      graph_internal,
      queries,
      neighbors,
      distances,
      sample_filter,
      index.metric());
  } else if (auto* empty_dset = dynamic_cast<const empty_dataset<ds_idx_type>*>(&index.data());
             empty_dset != nullptr) {
    // Forgot to add a dataset.
    RAFT_FAIL(
      "Attempted to search without a dataset. Please call index.update_dataset(...) first.");
  } else {
    // This is a logic error.
    RAFT_FAIL("Unrecognized dataset format");
  }

  static_assert(std::is_same_v<DistanceT, float>,
                "only float distances are supported at the moment");
  float* dist_out          = distances.data_handle();
  const DistanceT* dist_in = distances.data_handle();
  // We're converting the data from T to DistanceT during distance computation
  // and divide the values by kDivisor. Here we restore the original scale.
  constexpr float kScale = spatial::knn::detail::utils::config<T>::kDivisor /
                           spatial::knn::detail::utils::config<DistanceT>::kDivisor;
  ivf::detail::postprocess_distances(dist_out,
                                     dist_in,
                                     index.metric(),
                                     distances.extent(0),
                                     distances.extent(1),
                                     kScale,
                                     true,
                                     resource::get_cuda_stream(res));
}
/** @} */  // end group cagra

}  // namespace raft::neighbors::cagra::detail
