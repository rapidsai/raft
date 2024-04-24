#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import warnings

from pylibraft.distance.distance_type cimport DistanceType

SUPPORTED_DISTANCES = {
    "sqeuclidean": DistanceType.L2Expanded,
    "euclidean": DistanceType.L2SqrtExpanded,
    "inner_product": DistanceType.InnerProduct,
    "cosine": DistanceType.CosineExpanded
}


def _get_metric(metric):
    if metric not in SUPPORTED_DISTANCES:
        if metric == "l2_expanded":
            warnings.warn("Using l2_expanded as a metric name is deprecated,"
                          " use sqeuclidean instead", FutureWarning)
            return DistanceType.L2Expanded

        raise ValueError("metric %s is not supported" % metric)
    return SUPPORTED_DISTANCES[metric]


cdef _get_metric_string(DistanceType metric):
    return {DistanceType.L2Expanded : "sqeuclidean",
            DistanceType.InnerProduct: "inner_product",
            DistanceType.L2SqrtExpanded: "euclidean",
            DistanceType.CosineExpanded: "cosine"}[metric]


def _check_input_array(cai, exp_dt, exp_rows=None, exp_cols=None):
    if cai.dtype not in exp_dt:
        raise TypeError("dtype %s not supported" % cai.dtype)

    if not cai.c_contiguous:
        raise ValueError("Row major input is expected")

    if exp_cols is not None and cai.shape[1] != exp_cols:
        raise ValueError("Incorrect number of columns, expected {} got {}"
                         .format(exp_cols, cai.shape[1]))

    if exp_rows is not None and cai.shape[0] != exp_rows:
        raise ValueError("Incorrect number of rows, expected {} , got {}"
                         .format(exp_rows, cai.shape[0]))
