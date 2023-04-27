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

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t
from libcpp cimport bool

import numpy as np

from pylibraft.common import auto_convert_output, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle
from pylibraft.common.input_validation import is_c_contiguous

from pylibraft.common.cpp.mdspan cimport (
    device_matrix_view,
    host_matrix_view,
    make_device_matrix_view,
    make_host_matrix_view,
    row_major,
)
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources
from pylibraft.common.mdspan cimport get_dmv_float, get_dmv_int64
from pylibraft.matrix.cpp.select_k cimport select_k as c_select_k


@auto_sync_handle
@auto_convert_output
def select_k(dataset, k=None, distances=None, indices=None, select_min=True,
             handle=None):
    """
    Selects the top k items from each row in a matrix


    Parameters
    ----------
    dataset : array interface compliant matrix, row-major layout,
        shape (n_rows, dim). Supported dtype [float]
    k : int
        Number of items to return for each row.  Optional if indices or
        distances arrays are given (in which case their second dimension
        is k).
    distances :  Optional array interface compliant matrix shape
                (n_rows, k), dtype float. If supplied,
                distances will be written here in-place. (default None)
    indices :  Optional array interface compliant matrix shape
                (n_rows, k), dtype int64_t. If supplied, neighbor
                indices will be written here in-place. (default None)
    select_min: : bool
        Whether to select the minimum or maximum K items

    {handle_docstring}

    Returns
    -------
    distances: array interface compliant object containing resulting distances
               shape (n_rows, k)

    indices: array interface compliant object containing resulting indices
             shape (n_rows, k)

    Examples
    --------

    >>> import cupy as cp

    >>> from pylibraft.matrix import select_k

    >>> n_features = 50
    >>> n_rows = 1000

    >>> queries = cp.random.random_sample((n_rows, n_features),
    ...                                   dtype=cp.float32)
    >>> k = 40
    >>> distances, ids = select_k(queries, k)
    >>> distances = cp.asarray(distances)
    >>> ids = cp.asarray(ids)
    """

    dataset_cai = cai_wrapper(dataset)

    if k is None:
        if indices is not None:
            k = cai_wrapper(indices).shape[1]
        elif distances is not None:
            k = cai_wrapper(distances).shape[1]
        else:
            raise ValueError("Argument k must be specified if both indices "
                             "and distances arg is None")

    n_rows = dataset.shape[0]
    if indices is None:
        indices = device_ndarray.empty((n_rows, k), dtype='int64')

    if distances is None:
        distances = device_ndarray.empty((n_rows, k), dtype='float32')

    distances_cai = cai_wrapper(distances)
    indices_cai = cai_wrapper(indices)

    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef optional[device_matrix_view[int64_t, int64_t, row_major]] in_idx

    if dataset_cai.dtype == np.float32:
        c_select_k(deref(handle_),
                   get_dmv_float(dataset_cai, check_shape=True),
                   in_idx,
                   get_dmv_float(distances_cai, check_shape=True),
                   get_dmv_int64(indices_cai, check_shape=True),
                   <bool>select_min)
    else:
        raise TypeError("dtype %s not supported" % dataset_cai.dtype)

    return distances, indices
