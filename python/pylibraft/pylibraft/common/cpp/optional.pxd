#
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


# We're still using cython v0.29.x - which doesn't have std::optional
# support. Include the minimal definition here as suggested by
# https://github.com/cython/cython/issues/3293#issuecomment-1223058101

cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass optional[T]:
        optional()
        optional& operator=[U](U&)
