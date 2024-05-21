#
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import io

import numpy as np

from cpython.buffer cimport PyBUF_FULL_RO, PyBuffer_Release, PyObject_GetBuffer
from cpython.object cimport PyObject
from cython.operator cimport dereference as deref
from libc.stddef cimport size_t
from libc.stdint cimport int8_t, int32_t, int64_t, uint8_t, uint32_t, uintptr_t
from libcpp cimport bool

from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    host_matrix_view,
    host_mdspan,
    make_device_matrix_view,
    make_host_matrix_view,
    matrix_extent,
    ostream,
    ostringstream,
    row_major,
    serialize_mdspan,
)
from pylibraft.common.handle cimport device_resources
from pylibraft.common.optional cimport make_optional, optional

from pylibraft.common import DeviceResources


def run_roundtrip_test_for_mdspan(X, fortran_order=False):
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise ValueError("Please call this function with a NumPy array with"
                         "2 dimensions")
    handle = DeviceResources()
    cdef device_resources * handle_ = \
        <device_resources *> <size_t> handle.getHandle()
    cdef ostringstream oss
    cdef Py_buffer buf
    PyObject_GetBuffer(X, &buf, PyBUF_FULL_RO)
    cdef uintptr_t buf_ptr = <uintptr_t>buf.buf
    if X.dtype == np.float32:
        if fortran_order:
            serialize_mdspan[float, matrix_extent[size_t], col_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[float, matrix_extent[size_t],
                                   col_major] &>
                make_host_matrix_view[float, size_t, col_major](
                    <float *>buf_ptr,
                    X.shape[0], X.shape[1]))
        else:
            serialize_mdspan[float, matrix_extent[size_t], row_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[float, matrix_extent[size_t],
                                   row_major]&>
                make_host_matrix_view[float, size_t, row_major](
                    <float *>buf_ptr,
                    X.shape[0], X.shape[1]))
    elif X.dtype == np.float64:
        if fortran_order:
            serialize_mdspan[double, matrix_extent[size_t], col_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[double, matrix_extent[size_t],
                                   col_major]&>
                make_host_matrix_view[double, size_t, col_major](
                    <double *>buf_ptr,
                    X.shape[0], X.shape[1]))
        else:
            serialize_mdspan[double, matrix_extent[size_t], row_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[double, matrix_extent[size_t],
                                   row_major]&>
                make_host_matrix_view[double, size_t, row_major](
                    <double *>buf_ptr,
                    X.shape[0], X.shape[1]))
    elif X.dtype == np.int32:
        if fortran_order:
            serialize_mdspan[int32_t, matrix_extent[size_t], col_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[int32_t, matrix_extent[size_t],
                                   col_major]&>
                make_host_matrix_view[int32_t, size_t, col_major](
                    <int32_t *>buf_ptr,
                    X.shape[0], X.shape[1]))
        else:
            serialize_mdspan[int32_t, matrix_extent[size_t], row_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[int32_t, matrix_extent[size_t],
                                   row_major]&>
                make_host_matrix_view[int32_t, size_t, row_major](
                    <int32_t *>buf_ptr,
                    X.shape[0], X.shape[1]))
    elif X.dtype == np.uint32:
        if fortran_order:
            serialize_mdspan[uint32_t, matrix_extent[size_t], col_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[uint32_t, matrix_extent[size_t],
                                   col_major]&>
                make_host_matrix_view[uint32_t, size_t, col_major](
                    <uint32_t *>buf_ptr,
                    X.shape[0], X.shape[1]))
        else:
            serialize_mdspan[uint32_t, matrix_extent[size_t], row_major](
                deref(handle_),
                <ostream&>oss,
                <const host_mdspan[uint32_t, matrix_extent[size_t],
                                   row_major]&>
                make_host_matrix_view[uint32_t, size_t, row_major](
                    <uint32_t *>buf_ptr,
                    X.shape[0], X.shape[1]))
    else:
        PyBuffer_Release(&buf)
        raise NotImplementedError()
    PyBuffer_Release(&buf)
    f = io.BytesIO(oss.str())
    X2 = np.load(f)
    assert np.all(X.shape == X2.shape)
    assert np.all(X == X2)


cdef device_matrix_view[float, int64_t, row_major] \
        get_dmv_float(cai, check_shape) except *:
    if cai.dtype != np.float32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[float, int64_t, row_major](
        <float*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[bool, int64_t, row_major] \
        get_dmv_bool(cai, check_shape) except *:
    if cai.dtype != np.bool_:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[bool, int64_t, row_major](
        <bool*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[uint8_t, int64_t, row_major] \
        get_dmv_uint8(cai, check_shape) except *:
    if cai.dtype != np.uint8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[uint8_t, int64_t, row_major](
        <uint8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[int8_t, int64_t, row_major] \
        get_dmv_int8(cai, check_shape) except *:
    if cai.dtype != np.int8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[int8_t, int64_t, row_major](
        <int8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[int64_t, int64_t, row_major] \
        get_dmv_int64(cai, check_shape) except *:
    if cai.dtype != np.int64:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[int64_t, int64_t, row_major](
        <int64_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[const_float, int64_t, row_major] \
        get_const_dmv_float(cai, check_shape) except *:
    if cai.dtype != np.float32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[const_float, int64_t, row_major](
        <const float*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[const_uint8_t, int64_t, row_major] \
        get_const_dmv_uint8(cai, check_shape) except *:
    if cai.dtype != np.uint8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[const_uint8_t, int64_t, row_major](
        <const uint8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef device_matrix_view[const_int8_t, int64_t, row_major] \
        get_const_dmv_int8(cai, check_shape) except *:
    if cai.dtype != np.int8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[const_int8_t, int64_t, row_major](
        <const int8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef optional[device_matrix_view[int64_t, int64_t, row_major]] \
        make_optional_view_int64(device_matrix_view[int64_t, int64_t, row_major]& dmv) except *:  # noqa: E501
    return make_optional[device_matrix_view[int64_t, int64_t, row_major]](dmv)


# todo(dantegd): we can unify and simplify this functions a little bit
# defining extra functions as-is is the quickest way to get what we need for
# cagra.pyx
cdef device_matrix_view[uint32_t, int64_t, row_major] \
        get_dmv_uint32(cai, check_shape) except *:
    if cai.dtype != np.uint32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_device_matrix_view[uint32_t, int64_t, row_major](
        <uint32_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[float, int64_t, row_major] \
        get_hmv_float(cai, check_shape) except *:
    if cai.dtype != np.float32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[float, int64_t, row_major](
        <float*><uintptr_t>cai.data, shape[0], shape[1])

cdef host_matrix_view[uint8_t, int64_t, row_major] \
        get_hmv_uint8(cai, check_shape) except *:
    if cai.dtype != np.uint8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[uint8_t, int64_t, row_major](
        <uint8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[int8_t, int64_t, row_major] \
        get_hmv_int8(cai, check_shape) except *:
    if cai.dtype != np.int8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[int8_t, int64_t, row_major](
        <int8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[int64_t, int64_t, row_major] \
        get_hmv_int64(cai, check_shape) except *:
    if cai.dtype != np.int64:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[int64_t, int64_t, row_major](
        <int64_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[uint32_t, int64_t, row_major] \
        get_hmv_uint32(cai, check_shape) except *:
    if cai.dtype != np.uint32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[uint32_t, int64_t, row_major](
        <uint32_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[uint64_t, int64_t, row_major] \
        get_hmv_uint64(cai, check_shape) except *:
    if cai.dtype != np.uint64:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[uint64_t, int64_t, row_major](
        <uint64_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[const_float, int64_t, row_major] \
        get_const_hmv_float(cai, check_shape) except *:
    if cai.dtype != np.float32:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[const_float, int64_t, row_major](
        <const float*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[const_uint8_t, int64_t, row_major] \
        get_const_hmv_uint8(cai, check_shape) except *:
    if cai.dtype != np.uint8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[const_uint8_t, int64_t, row_major](
        <const uint8_t*><uintptr_t>cai.data, shape[0], shape[1])


cdef host_matrix_view[const_int8_t, int64_t, row_major] \
        get_const_hmv_int8(cai, check_shape) except *:
    if cai.dtype != np.int8:
        raise TypeError("dtype %s not supported" % cai.dtype)
    if check_shape and len(cai.shape) != 2:
        raise ValueError("Expected a 2D array, got %d D" % len(cai.shape))
    shape = (cai.shape[0], cai.shape[1] if len(cai.shape) == 2 else 1)
    return make_host_matrix_view[const_int8_t, int64_t, row_major](
        <const_int8_t*><uintptr_t>cai.data, shape[0], shape[1])
