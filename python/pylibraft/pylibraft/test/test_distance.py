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

    def __init__(self, ndarray, order):
        self.ndarray_ = ndarray
        self.device_buffer_ = \
            rmm.DeviceBuffer.to_device(ndarray.ravel(order=order).tobytes())

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


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [100])
@pytest.mark.parametrize("metric", ["euclidean", "cityblock", "chebyshev",
                                    "canberra", "correlation", "hamming",
                                    "jensenshannon", "russellrao", "cosine",
                                    "sqeuclidean"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_distance(n_rows, n_cols, metric, order, dtype):
    input1 = np.random.random_sample((n_rows, n_cols))
    input1 = np.asarray(input1, order=order).astype(dtype)

    # RussellRao expects boolean arrays
    if metric == "russellrao":
        input1[input1 < 0.5] = 0
        input1[input1 >= 0.5] = 1

    # JensenShannon expects probability arrays
    elif metric == "jensenshannon":
        norm = np.sum(input1, axis=1)
        input1 = (input1.T / norm).T

    output = np.zeros((n_rows, n_rows), dtype=dtype)

    expected = cdist(input1, input1, metric)

    expected[expected <= 1e-5] = 0.0

    input1_device = TestDeviceBuffer(input1, order)
    output_device = TestDeviceBuffer(output, order)

    pairwise_distance(input1_device, input1_device, output_device, metric)
    actual = output_device.copy_to_host()

    actual[actual <= 1e-5] = 0.0

    assert np.allclose(expected, actual, rtol=1e-4)
