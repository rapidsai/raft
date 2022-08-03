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

import pytest
import numpy as np

import rmm

from pylibraft.random import rmat


class TestDeviceBuffer:

    def __init__(self, ndarray):
        self.ndarray_ = ndarray
        self.device_buffer_ = \
            rmm.DeviceBuffer.to_device(ndarray.ravel(order="C").tobytes())

    @property
    def __cuda_array_interface__(self):
        device_cai = self.device_buffer_.__cuda_array_interface__
        host_cai = self.ndarray_.__array_interface__.copy()
        host_cai["data"] = (device_cai["data"][0], device_cai["data"][1])
        return host_cai

    def copy_to_host(self):
        return np.frombuffer(self.device_buffer_.tobytes(),
                             dtype=self.ndarray_.dtype,
                             like=self.ndarray_)\
            .astype(self.ndarray_.dtype)\
            .reshape(self.ndarray_.shape)


@pytest.mark.parametrize("n_edges", [10000, 20000])
@pytest.mark.parametrize("r_scale", [16, 18])
@pytest.mark.parametrize("c_scale", [16, 18])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_rmat(n_edges, r_scale, c_scale, dtype):
    max_scale = max(r_scale, c_scale)
    theta = np.random.random_sample(max_scale * 4)
    for i in range(max_scale):
        a = theta[4 * i]
        b = theta[4 * i + 1]
        c = theta[4 * i + 2]
        d = theta[4 * i + 3]
        total = a + b + c + d
        theta[4 * i] = a / total
        theta[4 * i + 1] = b / total
        theta[4 * i + 2] = c / total
        theta[4 * i + 3] = d / total
    theta_device = TestDeviceBuffer(theta)
    out_buff = np.empty((n_edges, 2), dtype=dtype)
    output_device = TestDeviceBuffer(out_buff)
    rmat(output_device, theta_device, r_scale, c_scale, 12345)
    output = output_device.copy_to_host()
    # a more rigorous tests have been done at the c++ level
    assert np.all(output[:, 0] >= 0)
    assert np.all(output[:, 0] < 2**r_scale)
    assert np.all(output[:, 1] >= 0)
    assert np.all(output[:, 1] < 2**c_scale)
    rmat(output_device, theta_device, r_scale, c_scale, 12345)
    output1 = output_device.copy_to_host()
    assert np.all(np.equal(output, output1))
