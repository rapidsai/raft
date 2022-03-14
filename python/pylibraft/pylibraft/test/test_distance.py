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

from scipy.spatial.distance import cdist
import pytest
import numpy as np

import rmm

from pylibraft.distance import pairwise_distance


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


@pytest.mark.parametrize("n_rows", [10, 100, 1000])
@pytest.mark.parametrize("n_cols", [10, 100, 1000])
@pytest.mark.parametrize("dtype", [np.float32])
def test_distance(n_rows, n_cols, dtype):
    input1 = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    output = np.zeros((n_rows, n_rows), dtype=dtype)

    expected = cdist(input1, input1, "euclidean")

    input1_device = TestDeviceBuffer(input1)
    output_device = TestDeviceBuffer(output)

    pairwise_distance(input1_device, input1_device, output_device)
    actual = output_device.copy_to_host()

    assert np.allclose(expected, actual)
    # result = np.frombuffer(output_device.copy_to_host().tobytes(), dtype)

    # print(str(result.__array_interface__))
    #
    # print(str(result.dtype))
