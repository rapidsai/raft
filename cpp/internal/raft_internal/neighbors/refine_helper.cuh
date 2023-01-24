/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/random/rng.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

namespace raft::neighbors {

template <typename IdxT>
struct RefineInputs {
  IdxT n_queries;
  IdxT n_rows;
  IdxT dim;
  IdxT k;   // after refinement
  IdxT k0;  // initial k before refinement (k0 >= k).
  raft::distance::DistanceType metric;
  bool host_data;
};

/** Helper class to allocate arrays and generate input data for refinement test and benchmark. */
template <typename DataT, typename DistanceT, typename IdxT>
class RefineHelper {
 public:
  RefineHelper(const raft::handle_t& handle, RefineInputs<IdxT> params)
    : handle_(handle), stream_(handle.get_stream()), p(params)
  {
    raft::random::Rng r(1234ULL);

    dataset = raft::make_device_matrix<DataT, IdxT>(handle_, p.n_rows, p.dim);
    queries = raft::make_device_matrix<DataT, IdxT>(handle_, p.n_queries, p.dim);
    if constexpr (std::is_same<DataT, float>{}) {
      r.uniform(dataset.data_handle(), dataset.size(), DataT(-10.0), DataT(10.0), stream_);
      r.uniform(queries.data_handle(), queries.size(), DataT(-10.0), DataT(10.0), stream_);
    } else {
      r.uniformInt(dataset.data_handle(), dataset.size(), DataT(1), DataT(20), stream_);
      r.uniformInt(queries.data_handle(), queries.size(), DataT(1), DataT(20), stream_);
    }

    refined_distances = raft::make_device_matrix<DistanceT, IdxT>(handle_, p.n_queries, p.k);
    refined_indices   = raft::make_device_matrix<IdxT, IdxT>(handle_, p.n_queries, p.k);

    // Generate candidate vectors
    {
      candidates = raft::make_device_matrix<IdxT, IdxT>(handle_, p.n_queries, p.k0);
      rmm::device_uvector<DistanceT> distances_tmp(p.n_queries * p.k0, stream_);
      naive_knn<DistanceT, DataT, IdxT>(distances_tmp.data(),
                                        candidates.data_handle(),
                                        queries.data_handle(),
                                        dataset.data_handle(),
                                        p.n_queries,
                                        p.n_rows,
                                        p.dim,
                                        p.k0,
                                        p.metric,
                                        stream_);
      handle_.sync_stream(stream_);
    }

    if (p.host_data) {
      dataset_host    = raft::make_host_matrix<DataT, IdxT>(p.n_rows, p.dim);
      queries_host    = raft::make_host_matrix<DataT, IdxT>(p.n_queries, p.dim);
      candidates_host = raft::make_host_matrix<IdxT, IdxT>(p.n_queries, p.k0);

      raft::copy(dataset_host.data_handle(), dataset.data_handle(), dataset.size(), stream_);
      raft::copy(queries_host.data_handle(), queries.data_handle(), queries.size(), stream_);
      raft::copy(
        candidates_host.data_handle(), candidates.data_handle(), candidates.size(), stream_);

      refined_distances_host = raft::make_host_matrix<DistanceT, IdxT>(p.n_queries, p.k);
      refined_indices_host   = raft::make_host_matrix<IdxT, IdxT>(p.n_queries, p.k);
      handle_.sync_stream(stream_);
    }

    // Generate ground thruth for testing.
    {
      rmm::device_uvector<DistanceT> distances_dev(p.n_queries * p.k, stream_);
      rmm::device_uvector<IdxT> indices_dev(p.n_queries * p.k, stream_);
      naive_knn<DistanceT, DataT, IdxT>(distances_dev.data(),
                                        indices_dev.data(),
                                        queries.data_handle(),
                                        dataset.data_handle(),
                                        p.n_queries,
                                        p.n_rows,
                                        p.dim,
                                        p.k,
                                        p.metric,
                                        stream_);
      true_refined_distances_host.resize(p.n_queries * p.k);
      true_refined_indices_host.resize(p.n_queries * p.k);
      raft::copy(true_refined_indices_host.data(), indices_dev.data(), indices_dev.size(), stream_);
      raft::copy(
        true_refined_distances_host.data(), distances_dev.data(), distances_dev.size(), stream_);
      handle_.sync_stream(stream_);
    }
  }

 public:
  RefineInputs<IdxT> p;
  const raft::handle_t& handle_;
  rmm::cuda_stream_view stream_;

  raft::device_matrix<DataT, IdxT, row_major> dataset;
  raft::device_matrix<DataT, IdxT, row_major> queries;
  raft::device_matrix<IdxT, IdxT, row_major> candidates;  // Neighbor candidate indices
  raft::device_matrix<IdxT, IdxT, row_major> refined_indices;
  raft::device_matrix<DistanceT, IdxT, row_major> refined_distances;

  raft::host_matrix<DataT, IdxT, row_major> dataset_host;
  raft::host_matrix<DataT, IdxT, row_major> queries_host;
  raft::host_matrix<IdxT, IdxT, row_major> candidates_host;
  raft::host_matrix<IdxT, IdxT, row_major> refined_indices_host;
  raft::host_matrix<DistanceT, IdxT, row_major> refined_distances_host;

  std::vector<IdxT> true_refined_indices_host;
  std::vector<DistanceT> true_refined_distances_host;
};
}  // namespace raft::neighbors
