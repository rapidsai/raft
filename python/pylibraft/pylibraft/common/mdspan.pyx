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

from libc.stdint cimport uintptr_t

import numpy as np

from pylibraft.common.input_validation import is_c_contiguous

from pylibraft.common.mdspan cimport *


cdef _validate_array_interface(cai, expected_shape, ElementType * p) except +:
    """ checks an array interface dictionary to see if the shape, dtype, and
    strides match expectations """
    shape = cai["shape"]
    if len(shape) != expected_shape:
        raise ValueError(f"unexpected shape {shape} - "
                         f"expected {expected_shape} elements")

    dt = np.dtype(cai["typestr"])
    if dt.itemsize != sizeof(ElementType):
        raise ValueError(f"invalid dtype {dt}: has itemsize {dt.itemsize} but"
                         f" function expects {sizeof(ElementType)}")

    if not is_c_contiguous(cai, dt):
        raise ValueError("input must be c-contiguous")


cdef device_matrix_view[ElementType, int] device_matrix_view_from_array(
    arr, ElementType * p
) except +:
    """ Transform a CAI array to a device_matrix_view """
    # need to have the ElementType as one of the parameters, otherwise this
    # crashes the cython compiler =(
    cai = arr.__cuda_array_interface__
    _validate_array_interface(cai, 2, p)
    rows, cols = cai["shape"]
    ptr = <uintptr_t>cai["data"][0]
    return make_device_matrix_view(<ElementType*>ptr, <int>rows, <int>cols)


cdef device_matrix_view[const ElementType, int]
const_device_matrix_view_from_array(arr, ElementType * p) except +:
    """ Transform a CAI array to a device_matrix_view with a const element"""
    # I couldn't make cython accept a FusedType that distiguishes between a
    # const/non-const ElementType - meaning that we have some duplicated
    # logic from the device_matrix_view_from_array  function here
    cai = arr.__cuda_array_interface__
    _validate_array_interface(cai, 2, p)
    rows, cols = cai["shape"]
    ptr = <uintptr_t>cai["data"][0]
    return make_device_matrix_view(<const ElementType*>ptr,
                                   <int>rows, <int>cols)
