# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
import pytest

from pylibraft.common import DeviceResources, Stream, device_ndarray
from pylibraft.random import rmat

cupy = pytest.importorskip("cupy")


def generate_theta(r_scale, c_scale):
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
    theta_device = device_ndarray(theta)
    return theta, theta_device


@pytest.mark.parametrize("stream", [cupy.cuda.Stream().ptr, Stream()])
def test_handle_external_stream(stream):

    theta, theta_device = generate_theta(16, 16)
    out_buff = np.empty((1000, 2), dtype=np.int32)
    output_device = device_ndarray(out_buff)

    handle = DeviceResources()
    rmat(output_device, theta_device, 16, 16, 12345, handle=handle)
    handle.sync()

    with pytest.raises(ValueError):
        handle = DeviceResources(stream=1.0)
