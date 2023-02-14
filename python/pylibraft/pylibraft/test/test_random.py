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
import pytest

from pylibraft.common import DeviceResources, device_ndarray
from pylibraft.random import rmat


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


@pytest.mark.parametrize("n_edges", [10000, 20000])
@pytest.mark.parametrize("r_scale", [16, 18])
@pytest.mark.parametrize("c_scale", [16, 18])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_rmat(n_edges, r_scale, c_scale, dtype):
    theta, theta_device = generate_theta(r_scale, c_scale)
    out_buff = np.empty((n_edges, 2), dtype=dtype)
    output_device = device_ndarray(out_buff)

    handle = DeviceResources()
    rmat(output_device, theta_device, r_scale, c_scale, 12345, handle=handle)
    handle.sync()
    output = output_device.copy_to_host()
    # a more rigorous tests have been done at the c++ level
    assert np.all(output[:, 0] >= 0)
    assert np.all(output[:, 0] < 2**r_scale)
    assert np.all(output[:, 1] >= 0)
    assert np.all(output[:, 1] < 2**c_scale)
    rmat(output_device, theta_device, r_scale, c_scale, 12345, handle=handle)
    handle.sync()
    output1 = output_device.copy_to_host()
    assert np.all(np.equal(output, output1))


def test_rmat_exception():
    n_edges = 20000
    r_scale = c_scale = 16
    dtype = np.int32
    with pytest.raises(Exception) as exception:
        out_buff = np.empty((n_edges, 2), dtype=dtype)
        output_device = device_ndarray(out_buff)
        rmat(output_device, None, r_scale, c_scale, 12345)
        assert exception is not None
        assert exception.message == "'theta' cannot be None!"
    with pytest.raises(Exception) as exception:
        theta, theta_device = generate_theta(r_scale, c_scale)
        rmat(None, theta_device, r_scale, c_scale, 12345)
        assert exception is not None
        assert exception.message == "'out' cannot be None!"


def test_rmat_valueerror():
    n_edges = 20000
    r_scale = c_scale = 16
    with pytest.raises(ValueError) as exception:
        out_buff = np.empty((n_edges, 2), dtype=np.int16)
        output_device = device_ndarray(out_buff)
        theta, theta_device = generate_theta(r_scale, c_scale)
        rmat(output_device, theta_device, r_scale, c_scale, 12345)
        assert exception is not None
        assert "not supported" in exception.message
