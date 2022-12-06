/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <thrust/host_vector.h>

#include <raft/core/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/stats/dispersion.cuh>

namespace raft::cluster::kmeans::detail {

template <typename idx_t, typename value_t>
void find_k(const raft::handle_t& handle,
            raft::device_matrix_view<const value_t, idx_t> X,
            raft::device_matrix_view<value_t, idx_t> centroids,
            raft::device_vector_view<idx_t, idx_t> labels,
            raft::host_scalar_view<int> k_star,
            raft::host_scalar_view<value_t> residual,
            raft::host_scalar_view<idx_t> maxiter,
            idx_t kmax,
            idx_t kmin  = 1,
            value_t tol = 1e-3)
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

  auto clusterSizes = raft::make_device_vector<idx_t>(kmax);
  auto labels       = raft::make_device_vector<idx_t>(n);

  rmm::device_uvector<char> workspace(0, handle.get_stream());

  idx_t* clusterSizes_ptr = clusterSizes.data_handle();

  // Host memory
  auto results           = raft::make_host_vector<value_t>(kmax + 1);
  auto clusterDispersion = raft::make_host_vector<value_t>(kmax + 1);

  // Loop to find *best* k
  // Perform k-means in binary search
  int left   = kmin;  // must be at least 2
  int right  = kmax;  // int(floor(len(data)/2)) #assumption of clusters of size 2 at least
  int mid    = int(floor((right + left) / 2));
  int oldmid = mid;
  int tests  = 0;
  int iters  = 0;
  value_t objective[3];     // 0= left of mid, 1= right of mid
  if (left == 1) left = 2;  // at least do 2 clusters

  auto centroids_view =
    raft::make_device_matrix_view<const value_t, idx_t>(centroids.view(), left, d);
  raft::cluster::kmeans_fit_predict(
    handle, X.view(), std::nullptr, centroids_view, labels.view(), residual, maxiter);

  detail::countLabels(handle, labels.view(), clusterSizes.data_handle(), n, left, workspace);

  results[left] = *residual;

  clusterDispersion[left] =
    raft::stats::dispersion(handle, centroids_view, clusterSizes.view(), std::nullopt, n);
  // eval right edge0
  iters          = 0;
  results[right] = 1e20;
  while (results[right] > results[left] && tests < 3) {
    centroids_view =
      raft::make_device_matrix_view<const value_t, idx_t>(centroids.view(), right, d);
    raft::cluster::kmeans_fit_predict(
      handle, X.view(), std::nullptr, centroids_view, labels.view(), residual, maxiter);

    detail::countLabels(handle, labels.view(), clusterSizes.data_handle(), n, right, workspace);

    results[right] = *residual;
    clusterDispersion[right] =
      raft::stats::dispersion(handle, centroids_view, clusterSizes.view(), std::nullopt, n);
    tests += 1;
  }

  objective[0] = (n - left) / (left - 1) * clusterDispersion[left] / results[left];
  objective[1] = (n - right) / (right - 1) * clusterDispersion[right] / results[right];
  // printf(" L=%g,%g,R=%g,%g : resid,objectives\n", results[left], objective[0], results[right],
  // objective[1]); binary search
  while (left < right - 1) {
    results[mid] = 1e20;
    tests        = 0;
    iters        = 0;
    while (results[mid] > results[left] && tests < 3) {
      centroids_view =
        raft::make_device_matrix_view<const value_t, idx_t>(centroids.view(), mid, d);
      raft::cluster::kmeans_fit(handle, X.view(), std::nullptr, centroids_view, residual, maxiter);

      detail::countLabels(handle, labels.view(), clusterSizes.data_handle(), n, mid, workspace);

      results[mid] = *residual;
      clusterDispersion[mid] =
        raft::stats::dispersion(handle, centroids_view, clusterSizes.view(), std::nullopt, n);

      if (results[mid] > results[left] && (mid + 1) < right) {
        mid += 1;
        results[mid] = 1e20;
      } else if (results[mid] > results[left] && (mid - 1) > left) {
        mid -= 1;
        results[mid] = 1e20;
      }
      tests += 1;
    }
    // objective[0] =abs(results[left]-results[mid])  /(results[left]-minres);
    // objective[0] /= mid-left;
    // objective[1] =abs(results[mid] -results[right])/(results[mid]-minres);
    // objective[1] /= right-mid;

    // maximize Calinski-Harabasz Index, minimize resid/ cluster
    objective[0] = (n - left) / (left - 1) * clusterDispersion[left] / results[left];
    objective[1] = (n - right) / (right - 1) * clusterDispersion[right] / results[right];
    objective[2] = (n - mid) / (mid - 1) * clusterDispersion[mid] / results[mid];
    // yes, overwriting the above temporary results is what I want
    // printf(" L=%g M=%g R=%g : objectives\n", objective[0], objective[2], objective[1]);
    objective[0] = (objective[2] - objective[0]) / (mid - left);
    objective[1] = (objective[1] - objective[2]) / (right - mid);

    // printf(" L=%g,R=%g : d obj/ d k \n", objective[0], objective[1]);
    // printf(" left, mid, right, res_left, res_mid, res_right\n");
    // printf(" %d, %d, %d, %g, %g, %g\n", left, mid, right, results[left], results[mid],
    // results[right]);
    if (objective[0] > 0 && objective[1] < 0) {
      // our point is in the left-of-mid side
      right = mid;
    } else {
      left = mid;
    }
    oldmid = mid;
    mid    = int(floor((right + left) / 2));
  }
  *k_star      = right;
  objective[0] = (n - left) / (left - 1) * clusterDispersion[left] / results[left];
  objective[1] = (n - oldmid) / (oldmid - 1) * clusterDispersion[oldmid] / results[oldmid];
  // objective[0] =abs(results[left]-results[mid])  /(results[left]-minres);
  // objective[0] /= mid-left;
  // objective[1] =abs(results[mid] -results[right])/(results[mid]-minres);
  // objective[1] /= right-mid;
  if (objective[1] < objective[0]) { *k_star = left; }

  // if k_star isn't what we just ran, re-run to get correct centroids and dist data on return->
  // this saves memory
  if (*k_star != oldmid) {
    centroids_view =
      raft::make_device_matrix_view<const value_t, idx_t>(centroids.view(), *k_star, d);
    raft::cluster::kmeans_fit(handle, X.view(), std::nullptr, centroids_view, residual, maxiter);
  }

  *maxiter = iters;
}
}  // namespace raft::cluster::kmeans::detail