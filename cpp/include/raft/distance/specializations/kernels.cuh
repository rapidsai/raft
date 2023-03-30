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

#include <raft/distance/kernels/gram_matrix.cuh>
#include <raft/distance/kernels/kernel_matrices.cuh>

extern template class raft::distance::kernels::GramMatrixBase<double>;
extern template class raft::distance::kernels::GramMatrixBase<float>;

extern template class raft::distance::kernels::PolynomialKernel<double, int>;
extern template class raft::distance::kernels::PolynomialKernel<float, int>;

extern template class raft::distance::kernels::TanhKernel<double>;
extern template class raft::distance::kernels::TanhKernel<float>;

extern template class raft::distance::kernels::RBFKernel<double>;
extern template class raft::distance::kernels::RBFKernel<float>;