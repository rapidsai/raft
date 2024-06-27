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

#include "../common/util.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/refine.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <type_traits>

namespace raft::bench::ann {

inline raft::distance::DistanceType parse_metric_type(raft::bench::ann::Metric metric)
{
  if (metric == raft::bench::ann::Metric::kInnerProduct) {
    return raft::distance::DistanceType::InnerProduct;
  } else if (metric == raft::bench::ann::Metric::kEuclidean) {
    // Even for L2 expanded RAFT IVF Flat uses unexpanded formula
    return raft::distance::DistanceType::L2Expanded;
  } else {
    throw std::runtime_error("raft supports only metric type of inner product and L2");
  }
}

/** Report a more verbose error with a backtrace when OOM occurs on RMM side. */
inline auto rmm_oom_callback(std::size_t bytes, void*) -> bool
{
  auto cuda_status = cudaGetLastError();
  size_t free      = 0;
  size_t total     = 0;
  RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
  RAFT_FAIL(
    "Failed to allocate %zu bytes using RMM memory resource. "
    "NB: latest cuda status = %s, free memory = %zu, total memory = %zu.",
    bytes,
    cudaGetErrorName(cuda_status),
    free,
    total);
}

/**
 * This container keeps the part of raft state that should be shared among multiple copies of raft
 * handles (in different CPU threads).
 * An example of this is an RMM memory resource: if we had an RMM memory pool per thread, we'd
 * quickly run out of memory.
 */
class shared_raft_resources {
 public:
  using pool_mr_type  = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  using mr_type       = rmm::mr::failure_callback_resource_adaptor<pool_mr_type>;
  using large_mr_type = rmm::mr::managed_memory_resource;

  shared_raft_resources()
  try : orig_resource_{rmm::mr::get_current_device_resource()},
    pool_resource_(orig_resource_, 1024 * 1024 * 1024ull),
    resource_(&pool_resource_, rmm_oom_callback, nullptr), large_mr_() {
    rmm::mr::set_current_device_resource(&resource_);
  } catch (const std::exception& e) {
    auto cuda_status = cudaGetLastError();
    size_t free      = 0;
    size_t total     = 0;
    RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
    RAFT_FAIL(
      "Failed to initialize shared raft resources (NB: latest cuda status = %s, free memory = %zu, "
      "total memory = %zu): %s",
      cudaGetErrorName(cuda_status),
      free,
      total,
      e.what());
  }

  shared_raft_resources(shared_raft_resources&&)                       = delete;
  shared_raft_resources& operator=(shared_raft_resources&&)            = delete;
  shared_raft_resources(const shared_raft_resources& res)              = delete;
  shared_raft_resources& operator=(const shared_raft_resources& other) = delete;

  ~shared_raft_resources() noexcept { rmm::mr::set_current_device_resource(orig_resource_); }

  auto get_large_memory_resource() noexcept
  {
    return static_cast<rmm::mr::device_memory_resource*>(&large_mr_);
  }

 private:
  rmm::mr::device_memory_resource* orig_resource_;
  pool_mr_type pool_resource_;
  mr_type resource_;
  large_mr_type large_mr_;
};

/**
 * This struct is used by multiple raft benchmark wrappers. It serves as a thread-safe keeper of
 * shared and private GPU resources (see below).
 *
 * - Accessing the same `configured_raft_resources` from concurrent threads is not safe.
 * - Accessing the copies of `configured_raft_resources` from concurrent threads is safe.
 * - There must be at most one "original" `configured_raft_resources` at any time, but as many
 *   copies of it as needed (modifies the program static state).
 */
class configured_raft_resources {
 public:
  /**
   * This constructor has the shared state passed unmodified but creates the local state anew.
   * It's used by the copy constructor.
   */
  explicit configured_raft_resources(const std::shared_ptr<shared_raft_resources>& shared_res)
    : shared_res_{shared_res},
      res_{std::make_unique<raft::device_resources>(
        rmm::cuda_stream_view(get_stream_from_global_pool()))}
  {
    // set the large workspace resource to the raft handle, but without the deleter
    // (this resource is managed by the shared_res).
    raft::resource::set_large_workspace_resource(
      *res_,
      std::shared_ptr<rmm::mr::device_memory_resource>(shared_res_->get_large_memory_resource(),
                                                       raft::void_op{}));
  }

  /** Default constructor creates all resources anew. */
  configured_raft_resources() : configured_raft_resources{std::make_shared<shared_raft_resources>()}
  {
  }

  configured_raft_resources(configured_raft_resources&&);
  configured_raft_resources& operator=(configured_raft_resources&&);
  ~configured_raft_resources() = default;
  configured_raft_resources(const configured_raft_resources& res)
    : configured_raft_resources{res.shared_res_}
  {
  }
  configured_raft_resources& operator=(const configured_raft_resources& other)
  {
    this->shared_res_ = other.shared_res_;
    return *this;
  }

  operator raft::resources&() noexcept { return *res_; }
  operator const raft::resources&() const noexcept { return *res_; }

  /** Get the main stream */
  [[nodiscard]] auto get_sync_stream() const noexcept { return resource::get_cuda_stream(*res_); }

 private:
  /** The resources shared among multiple raft handles / threads. */
  std::shared_ptr<shared_raft_resources> shared_res_;
  /**
   * Until we make the use of copies of raft::resources thread-safe, each benchmark wrapper must
   * have its own copy of it.
   */
  std::unique_ptr<raft::device_resources> res_ = std::make_unique<raft::device_resources>();
};

inline configured_raft_resources::configured_raft_resources(configured_raft_resources&&) = default;
inline configured_raft_resources& configured_raft_resources::operator=(
  configured_raft_resources&&) = default;

/** A helper to refine the neighbors when the data is on device or on host. */
template <typename DatasetT, typename QueriesT, typename CandidatesT>
void refine_helper(const raft::resources& res,
                   DatasetT dataset,
                   QueriesT queries,
                   CandidatesT candidates,
                   int k,
                   AnnBase::index_type* neighbors,
                   float* distances,
                   raft::distance::DistanceType metric)
{
  using data_type    = typename DatasetT::value_type;
  using index_type   = AnnBase::index_type;
  using extents_type = index_type;  // device-side refine requires this

  static_assert(std::is_same_v<data_type, typename QueriesT::value_type>);
  static_assert(std::is_same_v<data_type, typename DatasetT::value_type>);
  static_assert(std::is_same_v<index_type, typename CandidatesT::value_type>);

  extents_type batch_size = queries.extent(0);
  extents_type dim        = queries.extent(1);
  extents_type k0         = candidates.extent(1);

  if (raft::get_device_for_address(dataset.data_handle()) >= 0) {
    auto dataset_device = raft::make_device_matrix_view<const data_type, extents_type>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));
    auto queries_device = raft::make_device_matrix_view<const data_type, extents_type>(
      queries.data_handle(), batch_size, dim);
    auto candidates_device = raft::make_device_matrix_view<const index_type, extents_type>(
      candidates.data_handle(), batch_size, k0);
    auto neighbors_device =
      raft::make_device_matrix_view<index_type, extents_type>(neighbors, batch_size, k);
    auto distances_device =
      raft::make_device_matrix_view<float, extents_type>(distances, batch_size, k);

    raft::neighbors::refine<index_type, data_type, float, extents_type>(res,
                                                                        dataset_device,
                                                                        queries_device,
                                                                        candidates_device,
                                                                        neighbors_device,
                                                                        distances_device,
                                                                        metric);
  } else {
    auto dataset_host = raft::make_host_matrix_view<const data_type, extents_type>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));

    if (raft::get_device_for_address(queries.data_handle()) >= 0) {
      // Queries & results are on the device

      auto queries_host    = raft::make_host_matrix<data_type, extents_type>(batch_size, dim);
      auto candidates_host = raft::make_host_matrix<index_type, extents_type>(batch_size, k0);
      auto neighbors_host  = raft::make_host_matrix<index_type, extents_type>(batch_size, k);
      auto distances_host  = raft::make_host_matrix<float, extents_type>(batch_size, k);

      auto stream = resource::get_cuda_stream(res);
      raft::copy(queries_host.data_handle(), queries.data_handle(), queries_host.size(), stream);
      raft::copy(
        candidates_host.data_handle(), candidates.data_handle(), candidates_host.size(), stream);

      raft::resource::sync_stream(res);  // wait for the queries and candidates
      raft::neighbors::refine<index_type, data_type, float, extents_type>(res,
                                                                          dataset_host,
                                                                          queries_host.view(),
                                                                          candidates_host.view(),
                                                                          neighbors_host.view(),
                                                                          distances_host.view(),
                                                                          metric);

      raft::copy(neighbors, neighbors_host.data_handle(), neighbors_host.size(), stream);
      raft::copy(distances, distances_host.data_handle(), distances_host.size(), stream);

    } else {
      // Queries & results are on the host - no device sync / copy needed

      auto queries_host = raft::make_host_matrix_view<const data_type, extents_type>(
        queries.data_handle(), batch_size, dim);
      auto candidates_host = raft::make_host_matrix_view<const index_type, extents_type>(
        candidates.data_handle(), batch_size, k0);
      auto neighbors_host =
        raft::make_host_matrix_view<index_type, extents_type>(neighbors, batch_size, k);
      auto distances_host =
        raft::make_host_matrix_view<float, extents_type>(distances, batch_size, k);

      raft::neighbors::refine<index_type, data_type, float, extents_type>(
        res, dataset_host, queries_host, candidates_host, neighbors_host, distances_host, metric);
    }
  }
}

}  // namespace raft::bench::ann
