#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool, nullptr

from pylibraft.common.handle cimport handle_t
from pylibraft.common.optional cimport optional
from pylibraft.common.mdspan cimport *

cimport pylibraft.cluster.kmeans_types as kmeans_types


cdef extern from "raft_distance/kmeans.hpp" \
        namespace "raft::cluster::kmeans::runtime" nogil:

    cdef void fit(
        const handle_t & handle,
        const kmeans_types.KMeansParams& params,
        device_matrix_view[const float, int] X,
        optional[device_vector_view[const float, int]] sample_weight,
        device_matrix_view[float, int] inertia,
        host_scalar_view[float, int] inertia,
        host_scalar_view[int, int] n_iter) except +

    cdef void fit(
        const handle_t & handle,
        const kmeans_types.KMeansParams& params,
        device_matrix_view[const double, int] X,
        optional[device_vector_view[const double, int]] sample_weight,
        device_matrix_view[double, int] inertia,
        host_scalar_view[double, int] inertia,
        host_scalar_view[int, int] n_iter) except +
