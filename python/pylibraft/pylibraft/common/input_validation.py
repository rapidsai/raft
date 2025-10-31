#
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import numpy as np


def do_dtypes_match(*cais):
    last_dtype = cais[0].__cuda_array_interface__["typestr"]
    for cai in cais:
        typestr = cai.__cuda_array_interface__["typestr"]
        if last_dtype != typestr:
            return False
        last_dtype = typestr
    return True


def do_rows_match(*cais):
    last_row = cais[0].__cuda_array_interface__["shape"][0]
    for cai in cais:
        rows = cai.__cuda_array_interface__["shape"][0]
        if last_row != rows:
            return False
        last_row = rows
    return True


def do_cols_match(*cais):
    last_col = cais[0].__cuda_array_interface__["shape"][1]
    for cai in cais:
        cols = cai.__cuda_array_interface__["shape"][1]
        if last_col != cols:
            return False
        last_col = cols
    return True


def do_shapes_match(*cais):
    last_shape = cais[0].__cuda_array_interface__["shape"]
    for cai in cais:
        shape = cai.__cuda_array_interface__["shape"]
        if last_shape != shape:
            return False
        last_shape = shape
    return True


def is_c_contiguous(cai):
    """
    Checks whether an array is C contiguous.

    Parameters
    ----------
    cai : CUDA array interface

    """
    dt = np.dtype(cai["typestr"])
    return (
        "strides" not in cai
        or cai["strides"] is None
        or cai["strides"][1] == dt.itemsize
    )
