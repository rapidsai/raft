#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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


class ai_wrapper:
    """
    Simple wrapper around a array interface object to reduce
    boilerplate for extracting common information from the underlying
    dictionary.
    """

    def __init__(self, ai_arr):
        """
        Constructor accepts an array interface compliant array

        Parameters
        ----------
        ai_arr : array interface array
        """
        self.ai_ = ai_arr.__array_interface__
        self.from_cai = False

    @property
    def dtype(self):
        """
        Returns the dtype of the underlying array interface
        """
        return np.dtype(self.ai_["typestr"])

    @property
    def shape(self):
        """
        Returns the shape of the underlying array interface
        """
        return self.ai_["shape"]

    @property
    def c_contiguous(self):
        """
        Returns whether the underlying array interface has
        c-ordered (row-major) layout
        """
        return input_validation.is_c_contiguous(self.ai_)

    @property
    def f_contiguous(self):
        """
        Returns whether the underlying array interface has
        f-ordered (column-major) layout
        """
        return not input_validation.is_c_contiguous(self.ai_)

    @property
    def data(self):
        """
        Returns the data pointer of the underlying array interface
        """
        return self.ai_["data"][0]

    def validate_shape_dtype(self, expected_dims=None, expected_dtype=None):
        """Checks to see if the shape, dtype, and strides match expectations"""
        if expected_dims is not None and len(self.shape) != expected_dims:
            raise ValueError(
                f"unexpected shape {self.shape} - "
                f"expected {expected_dims} dimensions"
            )

        if expected_dtype is not None and self.dtype != expected_dtype:
            raise ValueError(
                f"invalid dtype {self.dtype}: expected " f"{expected_dtype}"
            )

        if not self.c_contiguous:
            raise ValueError("input must be c-contiguous")
