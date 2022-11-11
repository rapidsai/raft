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


class TestDeviceBuffer:
    def __init__(self, ndarray, order):

        self.ndarray_ = ndarray
        self.device_buffer_ = rmm.DeviceBuffer.to_device(
            ndarray.ravel(order=order).tobytes()
        )

    @property
    def __cuda_array_interface__(self):
        device_cai = self.device_buffer_.__cuda_array_interface__
        host_cai = self.ndarray_.__array_interface__.copy()
        host_cai["data"] = (device_cai["data"][0], device_cai["data"][1])

        return host_cai

    def copy_to_host(self):
        return (
            np.frombuffer(
                self.device_buffer_.tobytes(),
                dtype=self.ndarray_.dtype,
                like=self.ndarray_,
            )
            .astype(self.ndarray_.dtype)
            .reshape(self.ndarray_.shape)
        )
