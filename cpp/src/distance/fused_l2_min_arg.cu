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

#include <raft/distance/fused_l2_nn.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/specializations.cuh>
#include <raft/core/device_mdarray.hpp>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <raft/core/kvp.hpp>
#include <raft/core/handle.hpp>

namespace raft::distance::runtime {

    template <typename IndexT, typename DataT>
    struct KeyValueIndexOp {
        __host__ __device__ __forceinline__ IndexT
        operator()(const raft::KeyValuePair<IndexT, DataT>& a) const
        {
            printf("%d, %f\n", a.key, a.value);
            return a.key;
        }
    };

template<typename value_t, typename idx_t>
    void compute_fused_l2_nn_min_arg(
            raft::handle_t const& handle,
            idx_t* min,
            const value_t* x,
            const value_t* y,
            idx_t m,
            idx_t n,
            idx_t k,
            bool sqrt) {
        rmm::device_uvector<int> workspace(m, handle.get_stream());
        auto kvp = raft::make_device_vector<raft::KeyValuePair<idx_t, value_t>>(handle, m);

        rmm::device_uvector<value_t> x_norms(m, handle.get_stream());
        rmm::device_uvector<value_t> y_norms(n, handle.get_stream());
        raft::linalg::rowNorm(x_norms.data(), x, k, m, raft::linalg::L2Norm, true, handle.get_stream());
        raft::linalg::rowNorm(y_norms.data(), y, k, n, raft::linalg::L2Norm, true, handle.get_stream());

        fusedL2NNMinReduce(kvp.data_handle(), x, y, x_norms.data(), y_norms.data(), m, n, k, (void*)workspace.data(), sqrt, true, handle.get_stream());

        raft::print_device_vector("x", x, m*k, std::cout);
        raft::print_device_vector("y", y, n*k, std::cout);

    raft::print_device_vector("x_norms", x_norms.data(), m, std::cout);
    raft::print_device_vector("y_norms", y_norms.data(), n, std::cout);

    KeyValueIndexOp<idx_t, value_t> conversion_op;
        thrust::transform(handle.get_thrust_policy(), kvp.data_handle(), kvp.data_handle()+m, min, conversion_op);
        handle.sync_stream();
        raft::print_device_vector("min", min, m, std::cout);
    }

    /**
 * @brief Wrapper around fusedL2NN with minimum reduction operators.
 *
 * fusedL2NN cannot be compiled in the distance library due to the lambda
 * operators, so this wrapper covers the most common case (minimum).
 * This should be preferred to the more generic API when possible, in order to
 * reduce compilation times for users of the shared library.
 * @param[in] handle         raft handle
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  xn            L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in]  yn            L2 squared norm of `y`. Length = `n`. (on device)
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 */
    void fused_l2_nn_min_arg(
            raft::handle_t const& handle,
            int* min,
            const float* x,
            const float* y,
            int m,
            int n,
            int k,
            bool sqrt) {

        compute_fused_l2_nn_min_arg<float, int>(handle, min, x, y, m, n, k, sqrt);
    }

    void fused_l2_nn_min_arg(
            raft::handle_t const& handle,
            int* min,
            const double* x,
            const double* y,
            int m,
            int n,
            int k,
            bool sqrt) {

        compute_fused_l2_nn_min_arg<double, int>(handle, min, x, y, m, n, k, sqrt);
}


} // end namespace raft::distance::runtime