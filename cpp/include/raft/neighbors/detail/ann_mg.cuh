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

#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/comms/std_comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/ann_types.hpp>
#include <raft/neighbors/ann_mg_types.hpp>

#undef RAFT_EXPLICIT_INSTANTIATE_ONLY
#include <raft/neighbors/brute_force.cuh>
#define RAFT_EXPLICIT_INSTANTIATE_ONLY

#include <raft_runtime/neighbors/ivf_flat.hpp>
#include <raft/neighbors/ivf_flat_serialize.cuh>

#include <raft_runtime/neighbors/ivf_pq.hpp>
#include <raft/neighbors/ivf_pq_serialize.cuh>

#include <raft_runtime/neighbors/cagra.hpp>
#include <raft/neighbors/cagra_serialize.cuh>


// Number of rows per batch (search on shards)
#define N_ROWS_PER_BATCH 3000

namespace raft::neighbors::mg::detail {
using namespace raft::neighbors;

template <typename AnnIndexType, typename T, typename IdxT>
class ann_interface {
 public:
  void build(raft::resources const& handle,
             const ann::index_params* index_params,
             raft::host_matrix_view<const T, IdxT, row_major> h_index_dataset)
  {
    IdxT n_rows = h_index_dataset.extent(0);
    IdxT n_dims = h_index_dataset.extent(1);
    auto d_index_dataset = raft::make_device_matrix<T, IdxT, row_major>(handle, n_rows, n_dims);
    raft::copy(d_index_dataset.data_handle(), h_index_dataset.data_handle(), n_rows * n_dims, resource::get_cuda_stream(handle));
    raft::device_matrix_view<const T, IdxT, row_major> d_index_dataset_view = raft::make_device_matrix_view<const T, IdxT, row_major>(d_index_dataset.data_handle(), n_rows, n_dims);

    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      index_.emplace(std::move(raft::runtime::neighbors::ivf_flat::build(
        handle, *static_cast<const ivf_flat::index_params*>(index_params), d_index_dataset_view)));
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      index_.emplace(std::move(raft::runtime::neighbors::ivf_pq::build(
        handle, *static_cast<const ivf_pq::index_params*>(index_params), d_index_dataset_view)));
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      index_.emplace(std::move(raft::runtime::neighbors::cagra::build(
        handle, *static_cast<const cagra::index_params*>(index_params), d_index_dataset_view)));
    }
    resource::sync_stream(handle);
  }

  void extend(raft::resources const& handle,
              raft::host_matrix_view<const T, IdxT, row_major> h_new_vectors,
              std::optional<raft::host_vector_view<const IdxT, IdxT>> h_new_indices)
  {
    IdxT n_rows = h_new_vectors.extent(0);
    IdxT n_dims = h_new_vectors.extent(1);
    auto d_new_vectors = raft::make_device_matrix<T, IdxT, row_major>(handle, n_rows, n_dims);
    raft::copy(d_new_vectors.data_handle(), h_new_vectors.data_handle(), n_rows * n_dims, resource::get_cuda_stream(handle));
    raft::device_matrix_view<const T, IdxT, row_major> d_new_vectors_view = \
      raft::make_device_matrix_view<const T, IdxT, row_major>(d_new_vectors.data_handle(), n_rows, n_dims);

    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices_opt = std::nullopt;
    if (h_new_indices.has_value()) {
      auto d_new_indices = raft::make_device_vector<IdxT, IdxT>(handle, n_rows);
      raft::copy(d_new_indices.data_handle(), h_new_indices.value().data_handle(), n_rows, resource::get_cuda_stream(handle));
      auto d_new_indices_view = raft::device_vector_view<const IdxT, IdxT>(d_new_indices.data_handle(), n_rows);
      new_indices_opt = std::move(d_new_indices_view);
    }

    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      index_.emplace(std::move(raft::runtime::neighbors::ivf_flat::extend(
        handle, d_new_vectors_view, new_indices_opt, index_.value())));
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      index_.emplace(std::move(raft::runtime::neighbors::ivf_pq::extend(
        handle, d_new_vectors_view, new_indices_opt, index_.value())));
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      RAFT_FAIL("CAGRA does not implement the extend method");
    }
    resource::sync_stream(handle);
  }

  void search(raft::resources const& handle,
              const ann::search_params* search_params,
              raft::host_matrix_view<const T, IdxT, row_major> h_query_dataset,
              raft::device_matrix_view<IdxT, IdxT, row_major> d_neighbors,
              raft::device_matrix_view<float, IdxT, row_major> d_distances) const
  {
    IdxT n_rows = h_query_dataset.extent(0);
    IdxT n_dims = h_query_dataset.extent(1);
    auto d_query_dataset = raft::make_device_matrix<T, IdxT, row_major>(handle, n_rows, n_dims);
    raft::copy(d_query_dataset.data_handle(), h_query_dataset.data_handle(), n_rows * n_dims, resource::get_cuda_stream(handle));

    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, int64_t>>::value) {
      raft::runtime::neighbors::ivf_flat::search(handle,
                                                 *reinterpret_cast<const ivf_flat::search_params*>(search_params),
                                                 index_.value(),
                                                 d_query_dataset.view(),
                                                 d_neighbors,
                                                 d_distances);
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<int64_t>>::value) {
      raft::runtime::neighbors::ivf_pq::search(handle,
                                               *reinterpret_cast<const ivf_pq::search_params*>(search_params),
                                               index_.value(),
                                               d_query_dataset.view(),
                                               d_neighbors,
                                               d_distances);
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, uint32_t>>::value) {
      raft::runtime::neighbors::cagra::search(handle,
                                              *reinterpret_cast<const cagra::search_params*>(search_params),
                                              index_.value(),
                                              d_query_dataset.view(),
                                              d_neighbors,
                                              d_distances);
    }
    resource::sync_stream(handle);
  }

  void serialize(raft::resources const& handle,
                 std::ostream& os) const
   {
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      ivf_flat::serialize<T, IdxT>(handle, os, index_.value());
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      ivf_pq::serialize<IdxT>(handle, os, index_.value());
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      cagra::serialize<T, IdxT>(handle, os, index_.value());
    }
  }

  void deserialize(raft::resources const& handle,
                   std::istream& is)
  {
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      index_.emplace(std::move(ivf_flat::deserialize<T, IdxT>(handle, is)));
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      index_.emplace(std::move(ivf_pq::deserialize<IdxT>(handle, is)));
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      index_.emplace(std::move(cagra::deserialize<T, IdxT>(handle, is)));
    }
  }

  void deserialize(raft::resources const& handle,
                   const std::string& filename)
  {
    std::ifstream is(filename, std::ios::in | std::ios::binary);
    if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      index_.emplace(std::move(ivf_flat::deserialize<T, IdxT>(handle, is)));
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      index_.emplace(std::move(ivf_pq::deserialize<IdxT>(handle, is)));
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      index_.emplace(std::move(cagra::deserialize<T, IdxT>(handle, is)));
    }

    is.close();
  }

  const IdxT size() const
  {
    if constexpr (std::is_same<AnnIndexType, ivf_flat::index<T, IdxT>>::value) {
      return index_.value().size();
    } else if constexpr (std::is_same<AnnIndexType, ivf_pq::index<IdxT>>::value) {
      return index_.value().size();
    } else if constexpr (std::is_same<AnnIndexType, cagra::index<T, IdxT>>::value) {
      return index_.value().size();
    }
  }

 private:
  std::optional<AnnIndexType> index_;
};

template <typename AnnIndexType, typename T, typename IdxT>
class ann_mg_index {
 public:
  ann_mg_index(const std::vector<int>& dev_list, parallel_mode mode = SHARDING)
    : mode_(mode),
      root_rank_(0),
      num_ranks_(dev_list.size()),
      dev_ids_(dev_list),
      nccl_comms_(dev_list.size())
  {
    init_device_resources();
    init_nccl_clique();
  }

  // index deserialization and distribution
  ann_mg_index(const raft::resources& handle,
               const std::vector<int>& dev_list,
               const std::string& filename)
    : mode_(REPLICATION),
      root_rank_(0),
      num_ranks_(dev_list.size()),
      dev_ids_(dev_list),
      nccl_comms_(dev_list.size())
  {
      init_device_resources();
      init_nccl_clique();

      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        auto& ann_if = ann_interfaces_.emplace_back();
        ann_if.deserialize(dev_resources_[rank], filename);
      }
  }

  // MG index deserialization
  ann_mg_index(const raft::resources& handle,
               const std::string& filename) {
      std::ifstream is(filename, std::ios::in | std::ios::binary);
      if (!is) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

      mode_ = (raft::neighbors::mg::parallel_mode)deserialize_scalar<int>(handle, is);
      root_rank_ = 0;
      num_ranks_ = deserialize_scalar<int>(handle, is);
      dev_ids_.resize(num_ranks_);
      std::iota(std::begin(dev_ids_), std::end(dev_ids_), 0);
      nccl_comms_.resize(num_ranks_);

      init_device_resources();
      init_nccl_clique();

      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        auto& ann_if = ann_interfaces_.emplace_back();
        ann_if.deserialize(dev_resources_[rank], is);
      }

      is.close();
  }

  ann_mg_index(const ann_mg_index&)                    = delete;
  ann_mg_index(ann_mg_index&&)                         = default;
  auto operator=(const ann_mg_index&) -> ann_mg_index& = delete;
  auto operator=(ann_mg_index&&) -> ann_mg_index&      = default;

  void init_device_resources() {
    for (int rank = 0; rank < num_ranks_; rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
      dev_resources_.emplace_back();
    }
  }

  void init_nccl_clique() {
    RAFT_NCCL_TRY(ncclCommInitAll(nccl_comms_.data(), num_ranks_, dev_ids_.data()));
    for (int rank = 0; rank < num_ranks_; rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
      raft::comms::build_comms_nccl_only(&dev_resources_[rank], nccl_comms_[rank], num_ranks_, rank);
    }
  }

  void destroy_nccl_clique() {
    for (int rank = 0; rank < num_ranks_; rank++) {
      cudaSetDevice(dev_ids_[rank]);
      ncclCommDestroy(nccl_comms_[rank]);
    }
  }

  void build(const ann::index_params* index_params,
             raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
  {
    if (mode_ == REPLICATION) {
      ann_interfaces_.resize(num_ranks_);

      #pragma omp parallel for
      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        auto& ann_if = ann_interfaces_[rank];
        ann_if.build(dev_resources_[rank], index_params, index_dataset);
        resource::sync_stream(dev_resources_[rank]);
      }
      #pragma omp barrier
    } else if (mode_ == SHARDING) {
      IdxT n_rows           = index_dataset.extent(0);
      IdxT n_cols           = index_dataset.extent(1);

      ann_interfaces_.resize(num_ranks_);

      #pragma omp parallel for
      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        IdxT n_rows_per_shard  = raft::ceildiv(n_rows, (IdxT)num_ranks_);
        IdxT offset            = rank * n_rows_per_shard;
        n_rows_per_shard       = std::min(n_rows_per_shard, n_rows - offset);
        const T* partition_ptr = index_dataset.data_handle() + (offset * n_cols);
        auto partition         = raft::make_host_matrix_view<const T, IdxT, row_major>(
          partition_ptr, n_rows_per_shard, n_cols);
        auto& ann_if = ann_interfaces_[rank];
        ann_if.build(dev_resources_[rank], index_params, partition);
        resource::sync_stream(dev_resources_[rank]);
      }
      #pragma omp barrier
    }
    set_current_device_to_root_rank();
  }

  void extend(raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
              std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)
  {
    IdxT n_rows = new_vectors.extent(0);
    if (mode_ == REPLICATION) {
      #pragma omp parallel for
      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        auto& ann_if = ann_interfaces_[rank];
        ann_if.extend(dev_resources_[rank], new_vectors, new_indices);
        resource::sync_stream(dev_resources_[rank]);
      }
      #pragma omp barrier
    } else if (mode_ == SHARDING) {
      IdxT n_cols           = new_vectors.extent(1);
      #pragma omp parallel for
      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        IdxT n_rows_per_shard    = raft::ceildiv(n_rows, (IdxT)num_ranks_);
        IdxT offset              = rank * n_rows_per_shard;
        n_rows_per_shard         = std::min(n_rows_per_shard, n_rows - offset);
        const T* new_vectors_ptr = new_vectors.data_handle() + (offset * n_cols);
        auto new_vectors_part    = raft::make_host_matrix_view<const T, IdxT, row_major>(
          new_vectors_ptr, n_rows_per_shard, n_cols);

        std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices_part = std::nullopt;
        if (new_indices.has_value()) {
          const IdxT* new_indices_ptr = new_indices.value().data_handle() + offset;
          new_indices_part = raft::make_host_vector_view<const IdxT, IdxT>(new_indices_ptr, n_rows_per_shard);
        }
        auto& ann_if = ann_interfaces_[rank];
        ann_if.extend(dev_resources_[rank], new_vectors_part, new_indices_part);
        resource::sync_stream(dev_resources_[rank]);
      }
      #pragma omp barrier
    }
    set_current_device_to_root_rank();
  }

  void search(const ann::search_params* search_params,
              raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
              raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
              raft::host_matrix_view<float, IdxT, row_major> distances) const
  {
    if (mode_ == REPLICATION) {
      IdxT n_rows           = query_dataset.extent(0);
      IdxT n_cols           = query_dataset.extent(1);
      IdxT n_neighbors      = neighbors.extent(1);

      #pragma omp parallel for
      for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        IdxT n_rows_per_shard   = raft::ceildiv(n_rows, (IdxT)num_ranks_);
        IdxT offset             = rank * n_rows_per_shard;
        IdxT query_offset       = offset * n_cols;
        IdxT output_offset      = offset * n_neighbors;
        n_rows_per_shard        = std::min(n_rows_per_shard, n_rows - offset);
        auto query_partition = raft::make_host_matrix_view<const T, IdxT, row_major>(
          query_dataset.data_handle() + query_offset, n_rows_per_shard, n_cols);

        auto& handle = dev_resources_[rank];
        auto d_neighbors = raft::make_device_matrix<IdxT, IdxT, row_major>(handle, n_rows_per_shard, n_neighbors);
        auto d_distances = raft::make_device_matrix<float, IdxT, row_major>(handle, n_rows_per_shard, n_neighbors);

        auto& ann_if = ann_interfaces_[rank];
        ann_if.search(handle,
                      search_params,
                      query_partition,
                      d_neighbors.view(),
                      d_distances.view());

        raft::copy(neighbors.data_handle() + output_offset,
                   d_neighbors.data_handle(),
                   n_rows_per_shard * n_neighbors,
                   resource::get_cuda_stream(handle));
        raft::copy(distances.data_handle() + output_offset,
                   d_distances.data_handle(),
                   n_rows_per_shard * n_neighbors,
                   resource::get_cuda_stream(handle));
        resource::sync_stream(handle);
      }
      #pragma omp barrier
    } else if (mode_ == SHARDING) {
      IdxT n_rows      = query_dataset.extent(0);
      IdxT n_cols      = query_dataset.extent(1);
      IdxT n_neighbors = neighbors.extent(1);

      IdxT n_batches   = raft::ceildiv(n_rows, (IdxT)N_ROWS_PER_BATCH);

      const auto& root_handle = set_current_device_to_root_rank();
      auto in_neighbors       = raft::make_device_matrix<IdxT, IdxT, row_major>(
        root_handle, num_ranks_ * N_ROWS_PER_BATCH, n_neighbors);
      auto in_distances = raft::make_device_matrix<float, IdxT, row_major>(
        root_handle, num_ranks_ * N_ROWS_PER_BATCH, n_neighbors);
      auto out_neighbors = raft::make_device_matrix<IdxT, IdxT, row_major>(
        root_handle, N_ROWS_PER_BATCH, n_neighbors);
      auto out_distances = raft::make_device_matrix<float, IdxT, row_major>(
        root_handle, N_ROWS_PER_BATCH, n_neighbors);

      for (IdxT batch_idx = 0; batch_idx < n_batches; batch_idx++) {
        IdxT offset            = batch_idx * N_ROWS_PER_BATCH;
        IdxT query_offset      = offset * n_cols;
        IdxT output_offset     = offset * n_neighbors;
        IdxT n_rows_per_batches = std::min((IdxT)N_ROWS_PER_BATCH, n_rows - offset);
        auto query_partition = raft::make_host_matrix_view<const T, IdxT, row_major>(
          query_dataset.data_handle() + query_offset, n_rows_per_batches, n_cols);

        #pragma omp parallel for
        for (int rank = 0; rank < num_ranks_; rank++) {
          RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
          auto& handle = dev_resources_[rank];
          auto d_neighbors = raft::make_device_matrix<IdxT, IdxT, row_major>(handle, n_rows_per_batches, n_neighbors);
          auto d_distances = raft::make_device_matrix<float, IdxT, row_major>(handle, n_rows_per_batches, n_neighbors);

          auto& ann_if = ann_interfaces_[rank];
          ann_if.search(handle, search_params, query_partition, d_neighbors.view(), d_distances.view());

          RAFT_NCCL_TRY(ncclGroupStart());
          uint64_t batch_offset   = rank * n_rows_per_batches * n_neighbors;

          const auto& comms = resource::get_comms(handle);
          comms.device_send(d_neighbors.data_handle(),
                            n_rows_per_batches * n_neighbors,
                            root_rank_,
                            resource::get_cuda_stream(handle));
          comms.device_send(d_distances.data_handle(),
                            n_rows_per_batches * n_neighbors,
                            root_rank_,
                            resource::get_cuda_stream(handle));

          const auto& root_handle = set_current_device_to_root_rank();
          const auto& root_comms  = resource::get_comms(root_handle);
          root_comms.device_recv(in_neighbors.data_handle() + batch_offset,
                                 n_rows_per_batches * n_neighbors,
                                 rank,
                                 resource::get_cuda_stream(root_handle));
          root_comms.device_recv(in_distances.data_handle() + batch_offset,
                                 n_rows_per_batches * n_neighbors,
                                 rank,
                                 resource::get_cuda_stream(root_handle));
          RAFT_NCCL_TRY(ncclGroupEnd());
        }
        #pragma omp barrier

        auto in_neighbors_view  = raft::make_device_matrix_view<const IdxT, IdxT, row_major>(
          in_neighbors.data_handle(), num_ranks_ * n_rows_per_batches, n_neighbors);
        auto in_distances_view = raft::make_device_matrix_view<const float, IdxT, row_major>(
          in_distances.data_handle(), num_ranks_ * n_rows_per_batches, n_neighbors);
        auto out_neighbors_view = raft::make_device_matrix_view<IdxT, IdxT, row_major>(
          out_neighbors.data_handle(), n_rows_per_batches, n_neighbors);
        auto out_distances_view = raft::make_device_matrix_view<float, IdxT, row_major>(
          out_distances.data_handle(), n_rows_per_batches, n_neighbors);

        const auto& root_handle_ = set_current_device_to_root_rank();
        auto h_trans = std::vector<IdxT>(num_ranks_);
        IdxT translation_offset = 0;
        for (int rank = 0; rank < num_ranks_; rank++) {
          h_trans[rank] = translation_offset;
          translation_offset += ann_interfaces_[rank].size();
        }
        auto d_trans = raft::make_device_vector<IdxT, IdxT>(root_handle_, num_ranks_);
        raft::copy(d_trans.data_handle(), h_trans.data(), num_ranks_, resource::get_cuda_stream(root_handle_));
        auto translations = std::make_optional<raft::device_vector_view<IdxT, IdxT>>(d_trans.view());
        raft::neighbors::brute_force::knn_merge_parts<float, IdxT>(root_handle_,
                                                                   in_distances_view,
                                                                   in_neighbors_view,
                                                                   out_distances_view,
                                                                   out_neighbors_view,
                                                                   n_rows_per_batches,
                                                                   translations);

        raft::copy(neighbors.data_handle() + output_offset,
                   out_neighbors.data_handle(),
                   n_rows_per_batches * n_neighbors,
                   resource::get_cuda_stream(root_handle_));
        raft::copy(distances.data_handle() + output_offset,
                   out_distances.data_handle(),
                   n_rows_per_batches * n_neighbors,
                   resource::get_cuda_stream(root_handle_));
      }
    }

    set_current_device_to_root_rank();
  }

  void serialize(raft::resources const& handle,
                 const std::string& filename) const
  {
    std::ofstream of(filename, std::ios::out | std::ios::binary);
    if (!of) { RAFT_FAIL("Cannot open file %s", filename.c_str()); }

    serialize_scalar(handle, of, (int)mode_);
    serialize_scalar(handle, of, num_ranks_);
    for (int rank = 0; rank < num_ranks_; rank++) {
        RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[rank]));
        auto& ann_if = ann_interfaces_[rank];
        ann_if.serialize(dev_resources_[rank], of);
    }

    of.close();
    if (!of) { RAFT_FAIL("Error writing output %s", filename.c_str()); }
  }

  inline const raft::device_resources& set_current_device_to_root_rank() const
  {
    RAFT_CUDA_TRY(cudaSetDevice(dev_ids_[root_rank_]));
    return dev_resources_[root_rank_];
  }

 private:
  parallel_mode mode_;
  int root_rank_;
  int num_ranks_;
  std::vector<int> dev_ids_;
  std::vector<raft::device_resources> dev_resources_;
  std::vector<ann_interface<AnnIndexType, T, IdxT>> ann_interfaces_;
  std::vector<ncclComm_t> nccl_comms_;
};

template <typename T, typename IdxT>
ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> build(
  const raft::resources& handle,
  const ivf_flat::dist_index_params& index_params,
  raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
{
  ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> index(index_params.device_ids, index_params.mode);
  index.build(static_cast<const ann::index_params*>(&index_params), index_dataset);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> build(
  const raft::resources& handle,
  const ivf_pq::dist_index_params& index_params,
  raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
{
  ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> index(index_params.device_ids, index_params.mode);
  index.build(static_cast<const ann::index_params*>(&index_params), index_dataset);
  return index;
}

template <typename T, typename IdxT>
ann_mg_index<cagra::index<T, IdxT>, T, IdxT> build(
  const raft::resources& handle,
  const cagra::dist_index_params& index_params,
  raft::host_matrix_view<const T, IdxT, row_major> index_dataset)
{
  ann_mg_index<cagra::index<T, IdxT>, T, IdxT> index(index_params.device_ids, index_params.mode);
  index.build(static_cast<const ann::index_params*>(&index_params), index_dataset);
  return index;
}

template <typename T, typename IdxT>
void extend(const raft::resources& handle,
            ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)
{
  index.extend(new_vectors, new_indices);
}

template <typename T, typename IdxT>
void extend(const raft::resources& handle,
            ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices)
{
  index.extend(new_vectors, new_indices);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
            const ivf_flat::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances)
{
  index.search(
    static_cast<const ann::search_params*>(&search_params), query_dataset, neighbors, distances);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
            const ivf_pq::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances)
{
  index.search(
    static_cast<const ann::search_params*>(&search_params), query_dataset, neighbors, distances);
}

template <typename T, typename IdxT>
void search(const raft::resources& handle,
            const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
            const cagra::search_params& search_params,
            raft::host_matrix_view<const T, IdxT, row_major> query_dataset,
            raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::host_matrix_view<float, IdxT, row_major> distances)
{
  index.search(
    static_cast<const ann::search_params*>(&search_params), query_dataset, neighbors, distances);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  index.serialize(handle, filename);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  index.serialize(handle, filename);
}

template <typename T, typename IdxT>
void serialize(const raft::resources& handle,
               const ann_mg_index<cagra::index<T, IdxT>, T, IdxT>& index,
               const std::string& filename)
{
  index.serialize(handle, filename);
}

template <typename T, typename IdxT>
ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> deserialize_flat(const raft::resources& handle,
                                                                 const std::string& filename)
{
  return ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>(handle, filename);
}

template <typename T, typename IdxT>
ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> deserialize_pq(const raft::resources& handle,
                                                          const std::string& filename)
{
  return ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>(handle, filename);
}

template <typename T, typename IdxT>
ann_mg_index<cagra::index<T, IdxT>, T, IdxT> deserialize_cagra(const raft::resources& handle,
                                                               const std::string& filename)
{
  return ann_mg_index<cagra::index<T, IdxT>, T, IdxT>(handle, filename);
}

template <typename T, typename IdxT>
ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT> distribute_flat(const raft::resources& handle,
                                                                const std::vector<int>& dev_list,
                                                                const std::string& filename)
{
  return ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>(handle, dev_list, filename);
}

template <typename T, typename IdxT>
ann_mg_index<ivf_pq::index<IdxT>, T, IdxT> distribute_pq(const raft::resources& handle,
                                                         const std::vector<int>& dev_list,
                                                         const std::string& filename)
{
  return ann_mg_index<ivf_pq::index<IdxT>, T, IdxT>(handle, dev_list, filename);
}

template <typename T, typename IdxT>
ann_mg_index<cagra::index<T, IdxT>, T, IdxT> distribute_cagra(const raft::resources& handle,
                                                              const std::vector<int>& dev_list,
                                                              const std::string& filename)
{
  return ann_mg_index<cagra::index<T, IdxT>, T, IdxT>(handle, dev_list, filename);
}

}  // namespace raft::neighbors::mg::detail