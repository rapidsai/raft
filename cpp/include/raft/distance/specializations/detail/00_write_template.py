#!/usr/bin/env python3

# This template manages all files in this directory, apart from
# inner_product.cuh and kernels.cuh.


# NOTE: this template is not perfectly formatted. Use pre-commit to get
# everything in shape again.
start_template = """/*
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

#pragma once

#include <raft/distance/detail/distance.cuh>

namespace raft::distance::detail {

"""

extern_template = """
extern template void pairwise_matrix_instantiation_point<OpT,
                                                         IdxT,
                                                         DataT,
                                                         OutT,
                                                         FinopT>(
  OpT,
  pairwise_matrix_params<IdxT, DataT, OutT, FinopT>,
  cudaStream_t);
"""

end_template = """}  // namespace raft::distance::detail
"""

data_type_instances = [
    dict(
        DataT="float",
        AccT="float",
        OutT="float",
        IdxT="int",
    ),
    dict(
        DataT="double",
        AccT="double",
        OutT="double",
        IdxT="int",
    ),
]




op_instances = [
    dict(
        path_prefix="canberra",
        OpT="ops::canberra_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="correlation",
        OpT="ops::correlation_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="cosine",
        OpT="ops::cosine_distance_op<DataT, AccT, IdxT>",
        # cosine uses CUTLASS for SM80+
    ),
    dict(
        path_prefix="hamming_unexpanded",
        OpT="ops::hamming_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="hellinger_expanded",
        OpT="ops::hellinger_distance_op<DataT, AccT, IdxT>",
    ),
    # inner product is handled by cublas.
    dict(
        path_prefix="jensen_shannon",
        OpT="ops::jensen_shannon_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="kl_divergence",
        OpT="ops::kl_divergence_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="l1",
        OpT="ops::l1_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="l2_expanded",
        OpT="ops::l2_exp_distance_op<DataT, AccT, IdxT>",
        # L2 expanded uses CUTLASS for SM80+
    ),
    dict(
        path_prefix="l2_unexpanded",
        OpT="ops::l2_unexp_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="l_inf",
        OpT="ops::l_inf_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="lp_unexpanded",
        OpT="ops::lp_unexp_distance_op<DataT, AccT, IdxT>",
    ),
    dict(
        path_prefix="russel_rao",
        OpT="ops::russel_rao_distance_op<DataT, AccT, IdxT>",
    ),
]

def fill_in(s, template):
    for k, v in template.items():
        s = s.replace(k, v)
    return s

for op_instance in op_instances:
    path = fill_in("path_prefix.cuh", op_instance)
    with open(path, "w") as f:
        f.write(start_template)

        for data_type_instance in data_type_instances:
            op_data_instance = {
                k : fill_in(v, data_type_instance)
                for k, v in op_instance.items()
            }
            instance = {
                **op_data_instance,
                **data_type_instance,
                "FinopT": "raft::identity_op",
            }

            text = fill_in(extern_template, instance)

            f.write(text)

        f.write(end_template)
