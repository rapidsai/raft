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

#include <raft/cluster/detail/kmeans.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/dispersion.cuh>

#include <thrust/host_vector.h>

namespace raft::cluster::detail {

template <typename value_t, typename idx_t>
void compute_dispersion(raft::resources const& handle,
                        raft::device_matrix_view<const value_t, idx_t> X,
                        KMeansParams& params,
                        raft::device_matrix_view<value_t, idx_t> centroids_view,
                        raft::device_vector_view<idx_t> labels,
                        raft::device_vector_view<idx_t> clusterSizes,
                        rmm::device_uvector<char>& workspace,
                        raft::host_vector_view<value_t> clusterDispertionView,
                        raft::host_vector_view<value_t> resultsView,
                        raft::host_scalar_view<value_t> residual,
                        raft::host_scalar_view<idx_t> n_iter,
                        int val,
                        idx_t n,
                        idx_t d)
{
  auto centroids_const_view =
    raft::make_device_matrix_view<const value_t, idx_t>(centroids_view.data_handle(), val, d);

  idx_t* clusterSizes_ptr = clusterSizes.data_handle();
  auto cluster_sizes_view =
    raft::make_device_vector_view<const idx_t, idx_t>(clusterSizes_ptr, val);

  params.n_clusters = val;

  raft::cluster::detail::kmeans_fit_predict<value_t, idx_t>(
    handle, params, X, std::nullopt, std::make_optional(centroids_view), labels, residual, n_iter);

  detail::countLabels(handle, labels.data_handle(), clusterSizes.data_handle(), n, val, workspace);

  resultsView[val]           = residual[0];
  clusterDispertionView[val] = raft::stats::cluster_dispersion(
    handle, centroids_const_view, cluster_sizes_view, std::nullopt, n);
}

template <typename idx_t, typename value_t>
void find_k(raft::resources const& handle,
            raft::device_matrix_view<const value_t, idx_t> X,
            raft::host_scalar_view<idx_t> best_k,
            raft::host_scalar_view<value_t> residual,
            raft::host_scalar_view<idx_t> n_iter,
            idx_t kmax,
            idx_t kmin    = 1,
            idx_t maxiter = 100,
            value_t tol   = 1e-2)
{
  idx_t n = X.extent(0);
  idx_t d = X.extent(1);

  RAFT_EXPECTS(n >= 1, "n must be >= 1");
  RAFT_EXPECTS(d >= 1, "d must be >= 1");
  RAFT_EXPECTS(kmin >= 1, "kmin must be >= 1");
  RAFT_EXPECTS(kmax <= n, "kmax must be <= number of data samples in X");
  RAFT_EXPECTS(tol >= 0, "tolerance must be >= 0");
  RAFT_EXPECTS(maxiter >= 0, "maxiter must be >= 0");
  // Allocate memory
  // Device memory

  auto centroids    = raft::make_device_matrix<value_t, idx_t>(handle, kmax, X.extent(1));
  auto clusterSizes = raft::make_device_vector<idx_t>(handle, kmax);
  auto labels       = raft::make_device_vector<idx_t>(handle, n);

  rmm::device_uvector<char> workspace(0, resource::get_cuda_stream(handle));

  idx_t* clusterSizes_ptr = clusterSizes.data_handle();

  // Host memory
  auto results           = raft::make_host_vector<value_t>(kmax + 1);
  auto clusterDispersion = raft::make_host_vector<value_t>(kmax + 1);

  auto clusterDispertionView = clusterDispersion.view();
  auto resultsView           = results.view();

  // Loop to find *best* k
  // Perform k-means in binary search
  int left   = kmin;  // must be at least 2
  int right  = kmax;  // int(floor(len(data)/2)) #assumption of clusters of size 2 at least
  int mid    = ((unsigned int)left + (unsigned int)right) >> 1;
  int oldmid = mid;
  int tests  = 0;
  double objective[3];      // 0= left of mid, 1= right of mid
  if (left == 1) left = 2;  // at least do 2 clusters

  KMeansParams params;
  params.max_iter = maxiter;
  params.tol      = tol;

  auto centroids_view =
    raft::make_device_matrix_view<value_t, idx_t>(centroids.data_handle(), left, d);
  compute_dispersion<value_t, idx_t>(handle,
                                     X,
                                     params,
                                     centroids_view,
                                     labels.view(),
                                     clusterSizes.view(),
                                     workspace,
                                     clusterDispertionView,
                                     resultsView,
                                     residual,
                                     n_iter,
                                     left,
                                     n,
                                     d);

  // eval right edge0
  resultsView[right] = 1e20;
  while (resultsView[right] > resultsView[left] && tests < 3) {
    centroids_view =
      raft::make_device_matrix_view<value_t, idx_t>(centroids.data_handle(), right, d);
    compute_dispersion<value_t, idx_t>(handle,
                                       X,
                                       params,
                                       centroids_view,
                                       labels.view(),
                                       clusterSizes.view(),
                                       workspace,
                                       clusterDispertionView,
                                       resultsView,
                                       residual,
                                       n_iter,
                                       right,
                                       n,
                                       d);

    tests += 1;
  }

  objective[0] = (n - left) / (left - 1) * clusterDispertionView[left] / resultsView[left];
  objective[1] = (n - right) / (right - 1) * clusterDispertionView[right] / resultsView[right];
  while (left < right - 1) {
    resultsView[mid] = 1e20;
    tests            = 0;
    while (resultsView[mid] > resultsView[left] && tests < 3) {
      centroids_view =
        raft::make_device_matrix_view<value_t, idx_t>(centroids.data_handle(), mid, d);
      compute_dispersion<value_t, idx_t>(handle,
                                         X,
                                         params,
                                         centroids_view,
                                         labels.view(),
                                         clusterSizes.view(),
                                         workspace,
                                         clusterDispertionView,
                                         resultsView,
                                         residual,
                                         n_iter,
                                         mid,
                                         n,
                                         d);

      if (resultsView[mid] > resultsView[left] && (mid + 1) < right) {
        mid += 1;
        resultsView[mid] = 1e20;
      } else if (resultsView[mid] > resultsView[left] && (mid - 1) > left) {
        mid -= 1;
        resultsView[mid] = 1e20;
      }
      tests += 1;
    }

    // maximize Calinski-Harabasz Index, minimize resid/ cluster
    objective[0] = (n - left) / (left - 1) * clusterDispertionView[left] / resultsView[left];
    objective[1] = (n - right) / (right - 1) * clusterDispertionView[right] / resultsView[right];
    objective[2] = (n - mid) / (mid - 1) * clusterDispertionView[mid] / resultsView[mid];
    objective[0] = (objective[2] - objective[0]) / (mid - left);
    objective[1] = (objective[1] - objective[2]) / (right - mid);

    if (objective[0] > 0 && objective[1] < 0) {
      // our point is in the left-of-mid side
      right = mid;
    } else {
      left = mid;
    }
    oldmid = mid;
    mid    = ((unsigned int)right + (unsigned int)left) >> 1;
  }

  best_k[0]    = right;
  objective[0] = (n - left) / (left - 1) * clusterDispertionView[left] / resultsView[left];
  objective[1] = (n - oldmid) / (oldmid - 1) * clusterDispertionView[oldmid] / resultsView[oldmid];
  if (objective[1] < objective[0]) { best_k[0] = left; }

  // if best_k isn't what we just ran, re-run to get correct centroids and dist data on return->
  // this saves memory
  if (best_k[0] != oldmid) {
    auto centroids_view =
      raft::make_device_matrix_view<value_t, idx_t>(centroids.data_handle(), best_k[0], d);

    params.n_clusters = best_k[0];
    raft::cluster::detail::kmeans_fit_predict<value_t, idx_t>(handle,
                                                              params,
                                                              X,
                                                              std::nullopt,
                                                              std::make_optional(centroids_view),
                                                              labels.view(),
                                                              residual,
                                                              n_iter);
  }
}
}  // namespace raft::cluster::detail