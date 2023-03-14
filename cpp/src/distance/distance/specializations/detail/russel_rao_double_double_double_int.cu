/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>  // raft::identity_op
#include <raft/distance/detail/distance_ops/all_ops.cuh>

#include <raft/distance/detail/pairwise_matrix/dispatch.cuh>
#include <raft/util/arch.cuh>  // raft::arch::SM_compat_range

namespace raft::distance::detail {

template void pairwise_matrix_dispatch<ops::russel_rao_distance_op<double, double, int>,
                                       double,
                                       double,
                                       double,
                                       decltype(raft::identity_op()),
                                       int>(ops::russel_rao_distance_op<double, double, int>,
                                            int,
                                            int,
                                            int,
                                            const double*,
                                            const double*,
                                            const double*,
                                            const double*,
                                            double*,
                                            decltype(raft::identity_op()),
                                            cudaStream_t,
                                            bool);

}  // namespace raft::distance::detail
