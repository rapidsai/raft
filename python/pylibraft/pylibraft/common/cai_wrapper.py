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
import numpy as np

from pylibraft.common import input_validation


class cai_wrapper:
    """
    Simple wrapper around a CUDA array interface object to reduce
    boilerplate for extracting common information from the underlying
    dictionary.
    """

    def __init__(self, cai_arr):
        """
        Constructor accepts a CUDA array interface compliant array

        Parameters
        ----------
        cai_arr : CUDA array interface array
        """
        self.cai_ = cai_arr.__cuda_array_interface__

    @property
    def dtype(self):
        """
        Returns the dtype of the underlying CUDA array interface
        """
        return np.dtype(self.cai_["typestr"])

    @property
    def shape(self):
        """
        Returns the shape of the underlying CUDA array interface
        """
        return self.cai_["shape"]

    @property
    def c_contiguous(self):
        """
        Returns whether the underlying CUDA array interface has
        c-ordered (row-major) layout
        """
        return input_validation.is_c_contiguous(self.cai_)

    @property
    def f_contiguous(self):
        """
        Returns whether the underlying CUDA array interface has
        f-ordered (column-major) layout
        """
        return not input_validation.is_c_contiguous(self.cai_)

    @property
    def data(self):
        """
        Returns the data pointer of the underlying CUDA array interface
        """
        return self.cai_["data"][0]

    def validate(self, expected_dims=None, expected_dtype=None):
        """checks to see if the shape, dtype, and strides match expectations"""
        if expected_dims is not None and len(self.shape) != expected_dims:
            raise ValueError(
                f"unexpected shape {self.shape} - " f"expected {expected_dims} dimensions"
            )

        if expected_dtype is not None and self.dtype != expected_dtype:
            raise ValueError(
                f"invalid dtype {self.dtype}: expected " f"{expected_dtype}"
            )

        if not self.c_contiguous:
            raise ValueError("input must be c-contiguous")
