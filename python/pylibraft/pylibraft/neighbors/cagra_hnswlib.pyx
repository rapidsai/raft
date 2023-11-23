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
from libc.stdint cimport int8_t, uint8_t, uint32_t
from libcpp.string cimport string

cimport pylibraft.neighbors.cagra.cpp.c_cagra as c_cagra

from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.handle cimport device_resources

from pylibraft.common import DeviceResources


@auto_sync_handle
def save(filename, c_cagra.Index index, handle=None):
    """
    Saves the index to a file.

    Saving / loading the index is experimental. The serialization format is
    subject to change.

    Parameters
    ----------
    filename : string
        Name of the file.
    index : Index
        Trained CAGRA index.
    {handle_docstring}

    Examples
    --------
    >>> import cupy as cp
    >>> from pylibraft.common import DeviceResources
    >>> from pylibraft.neighbors import cagra
    >>> from pylibraft.neighbors import cagra_hnswlib
    >>> n_samples = 50000
    >>> n_features = 50
    >>> dataset = cp.random.random_sample((n_samples, n_features),
    ...                                   dtype=cp.float32)
    >>> # Build index
    >>> handle = DeviceResources()
    >>> index = cagra.build(cagra.IndexParams(), dataset, handle=handle)
    >>> # Serialize and deserialize the cagra index built
    >>> cagra_hnswlib.save("my_index.bin", index, handle=handle)
    """
    if not index.trained:
        raise ValueError("Index need to be built before saving it.")

    if handle is None:
        handle = DeviceResources()
    cdef device_resources* handle_ = \
        <device_resources*><size_t>handle.getHandle()

    cdef string c_filename = filename.encode('utf-8')

    cdef c_cagra.IndexFloat idx_float
    cdef c_cagra.IndexInt8 idx_int8
    cdef c_cagra.IndexUint8 idx_uint8

    cdef c_cagra.index[float, uint32_t] * c_index_float
    cdef c_cagra.index[int8_t, uint32_t] * c_index_int8
    cdef c_cagra.index[uint8_t, uint32_t] * c_index_uint8

    if index.active_index_type == "float32":
        idx_float = index
        c_index_float = \
            <c_cagra.index[float, uint32_t] *><size_t> idx_float.index
        c_cagra.serialize_to_hnswlib_file(
            deref(handle_), c_filename, deref(c_index_float))
    elif index.active_index_type == "byte":
        idx_int8 = index
        c_index_int8 = \
            <c_cagra.index[int8_t, uint32_t] *><size_t> idx_int8.index
        c_cagra.serialize_to_hnswlib_file(
            deref(handle_), c_filename, deref(c_index_int8))
    elif index.active_index_type == "ubyte":
        idx_uint8 = index
        c_index_uint8 = \
            <c_cagra.index[uint8_t, uint32_t] *><size_t> idx_uint8.index
        c_cagra.serialize_to_hnswlib_file(
            deref(handle_), c_filename, deref(c_index_uint8))
    else:
        raise ValueError(
            "Index dtype %s not supported" % index.active_index_type)
