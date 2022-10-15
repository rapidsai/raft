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

#include <raft/distance/detail/kernels/gram_matrix.cuh>
#include <raft/distance/detail/kernels/kernel_matrices.cuh>

extern template class raft::distance::kernels::detail::GramMatrixBase<double>;
extern template class raft::distance::kernels::detail::GramMatrixBase<float>;

extern template class raft::distance::kernels::detail::PolynomialKernel<double, int>;
extern template class raft::distance::kernels::detail::PolynomialKernel<float, int>;

extern template class raft::distance::kernels::detail::TanhKernel<double>;
extern template class raft::distance::kernels::detail::TanhKernel<float>;

// These are somehow missing a kernel definition which is causing a compile error
// extern template class raft::distance::kernels::detail::RBFKernel<double>;
// extern template class raft::distance::kernels::detail::RBFKernel<float>;