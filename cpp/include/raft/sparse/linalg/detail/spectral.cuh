/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/spectral/cluster_solvers.cuh>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/partition.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace raft {
namespace sparse {
namespace spectral {
namespace detail {

template <typename T>
void fit_embedding(raft::resources const& handle,
                   int* rows,
                   int* cols,
                   T* vals,
                   int nnz,
                   int n,
                   int n_components,
                   T* out,
                   unsigned long long seed = 1234567)
{
  auto stream = resource::get_cuda_stream(handle);
  rmm::device_uvector<int> src_offsets(n + 1, stream);
  rmm::device_uvector<int> dst_cols(nnz, stream);
  rmm::device_uvector<T> dst_vals(nnz, stream);
  convert::coo_to_csr(
    handle, rows, cols, vals, nnz, n, src_offsets.data(), dst_cols.data(), dst_vals.data());

  rmm::device_uvector<T> eigVals(n_components + 1, stream);
  rmm::device_uvector<T> eigVecs(n * (n_components + 1), stream);
  rmm::device_uvector<int> labels(n, stream);

  resource::sync_stream(handle, stream);

  /**
   * Raft spectral clustering
   */
  using index_type = int;
  using value_type = T;

  index_type* ro = src_offsets.data();
  index_type* ci = dst_cols.data();
  value_type* vs = dst_vals.data();

  raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{
    handle, ro, ci, vs, n, nnz};

  index_type neigvs       = n_components + 1;
  index_type maxiter      = 4000;  // default reset value (when set to 0);
  value_type tol          = 0.01;
  index_type restart_iter = 15 + neigvs;  // what cugraph is using

  raft::spectral::eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol};

  cfg.seed = seed;

  raft::spectral::lanczos_solver_t<index_type, value_type> eig_solver{cfg};

  // cluster computation here is irrelevant,
  // hence define a no-op such solver to
  // feed partition():
  //
  struct no_op_cluster_solver_t {
    using index_type_t = index_type;
    using size_type_t  = index_type;
    using value_type_t = value_type;

    std::pair<value_type_t, index_type_t> solve(raft::resources const& handle,
                                                size_type_t n_obs_vecs,
                                                size_type_t dim,
                                                value_type_t const* __restrict__ obs,
                                                index_type_t* __restrict__ codes) const
    {
      return std::make_pair<value_type_t, index_type_t>(0, 0);
    }
  };

  raft::spectral::partition(handle,
                            r_csr_m,
                            eig_solver,
                            no_op_cluster_solver_t{},
                            labels.data(),
                            eigVals.data(),
                            eigVecs.data());

  raft::copy<T>(out, eigVecs.data() + n, n * n_components, stream);

  RAFT_CUDA_TRY(cudaGetLastError());
}

};  // namespace detail
};  // namespace spectral
};  // namespace sparse
};  // namespace raft
