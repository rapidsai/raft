/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "brute_force_knn.cuh"
#include <raft/handle.hpp>

#include <raft/random/rng.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/sparse/convert/csr.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {


struct NNComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one &t1, const two &t2) {
    // sort first by each sample's reference landmark,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by closest neighbor,
    return thrust::get<1>(t1) < thrust::get<1>(t2);
  }
};



/**
 * Random ball cover algorithm uses the triangle inequality
 * (which can be used for any valid metric or distance
 * that follows the triangle inequality)
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
void random_ball_cover(const raft::handle_t &handle, const value_t *X,
                       value_idx m, value_idx n, int k, value_idx *inds,
                       value_t *dists) {
  /**
   * 1. Randomly sample sqrt(n) points from X
   */

  rmm::device_uvector<value_idx> R_knn_inds(k * m, handle.get_stream());
  rmm::device_uvector<value_t> R_knn_dists(k * m, handle.get_stream());

  value_idx n_samples = int(sqrt(m));

  rmm::device_uvector<value_idx> R_indices(n_samples, handle.get_stream());
  rmm::device_uvector<value_t> R(n_samples * n, handle.get_stream());
  raft::random::uniformInt(R_indices.data(), n_samples, 0, m-1, handle.get_stream());

  raft::matrix::copyRows(X, m, n, R.data(), R_indices.data(), n_samples,
                         handle.get_stream(), true);

  /**
   * 2. Perform knn = bfknn(X, R, k)
   */
  brute_force_knn_impl({X}, {m}, n, R.data(), n_samples, R_knn_inds.data(), R_knn_dists.data(), k,
                       handle.get_device_allocator(), handle.get_stream());

  /**
   * 3. Create L_r = knn[:,0].T (CSR)
   *
   * Slice closest neighboring R
   * Secondary sort by (R_knn_inds, R_knn_dists)
   */
  rmm::device_uvector<value_idx> R_1nn_inds(m, handle.get_stream());
  rmm::device_uvector<value_t> R_1nn_dists(m, handle.get_stream());
  rmm::device_uvector<value_idx> R_1nn_cols(m, handle.get_stream());

  raft::matrix::sliceMatrix(R_knn_inds.data(), m, k, R_1nn_inds.data(), 0,
                   1, m, 2, handle.get_stream());
  raft::matrix::sliceMatrix(R_knn_dists.data(), m, k, R_1nn_dists.data(), 0,https://arxiv.org/search/cs?searchtype=author&query=Domingos%2C+P
                            1, m, 2, handle.get_stream());

  thrust::sequence(thrust::cuda::par.on(handle.get_stream()), R_1nn_cols.data(),
                   R_1nn_cols.data()+m, 1);

  auto keys = thrust::make_zip_iterator(thrust::make_tuple(
    R_1nn_inds.data(), R_1nn_dists.data()));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(R_1nn_cols.data()));

  // group neighborhoods for each reference landmark and sort each group by distance
  thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + n_rows, vals,
                      NNComp());

  // convert to CSR for fast lookup
  rmm::device_uvector<value_idx> R_indptr(n_samples, handle.get_stream());
  raft::sparse::convert::sorted_coo_to_csr(R_1nn_inds.data(), m, R_indptr.data(), n_samples+1,
    handle.get_device_allocator(), handle.get_stream());

  /**
   * 4. Perform k-select over original KNN, using L_r to filter distances
   *
   * a. Map 1 row to each warp/block
   * b. Add closest R points to heap
   * c. Iterate through batches of R, having each thread in the warp load a set
   * of distances y from R and marking the distance to be computed between x, y only
   * if current knn[k].distance >= d(x_i, R_k) + d(R_k, y)
   */

}
};
};
};
};