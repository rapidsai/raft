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
import rmm

class device_ndarray:

    def __init__(self, np_ndarray):
        """
        Construct a raft.device_ndarray wrapper around a numpy.ndarray (on host)

        Parameters
        ----------
        ndarray : Any array that provides a valid __array_interface__ or __cuda_array_interface__
        """
        self.ndarray_ = np_ndarray
        order = "C" if self.c_contiguous else "F"
        self.device_buffer_ = rmm.DeviceBuffer.to_device(
            self.ndarray_.tobytes(order=order)
        )

    @property
    def c_contiguous(self):
        array_interface = self.ndarray_.__array_interface__
        strides = self.strides
        return strides is None or \
               array_interface["strides"][1] == self.dtype.itemsize

    @property
    def f_contiguous(self):
        return not self.c_contiguous

    @property
    def dtype(self):
        array_interface = self.ndarray_.__array_interface__
        return np.dtype(array_interface["typestr"])

    @property
    def shape(self):
        array_interface = self.ndarray_.__array_interface__
        return array_interface["shape"]

    @property
    def strides(self):
        array_interface = self.ndarray_.__array_interface__
        return None if "strides" not in array_interface else \
            array_interface["strides"]

    @property
    def __cuda_array_interface__(self):
        device_cai = self.device_buffer_.__cuda_array_interface__
        host_cai = self.ndarray_.__array_interface__.copy()
        host_cai["data"] = (device_cai["data"][0], device_cai["data"][1])

        return host_cai

    def copy_to_host(self):
        ret = (
            np.frombuffer(
                self.device_buffer_.tobytes(),
                dtype=self.dtype,
                like=self.ndarray_,
            )
            .astype(self.dtype)
        )
        ret = np.lib.stride_tricks.as_strided(ret, self.shape, self.strides)
        return ret
