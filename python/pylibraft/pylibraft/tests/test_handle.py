# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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

    with pytest.raises(TypeError):
        handle = DeviceResources(stream=1.0)


class CudaStreamProtocolObj:
    """A minimal object implementing the __cuda_stream__ protocol."""

    def __init__(self):
        self._s = Stream()

    def __cuda_stream__(self):
        return (0, self._s.get_ptr())


def test_stream_has_cuda_stream_protocol():
    """Stream class implements __cuda_stream__ protocol."""
    s = Stream()
    result = s.__cuda_stream__()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert result[0] == 0  # version must be 0
    assert result[1] == s.get_ptr()  # pointer must match get_ptr()
    assert result[1] != 0  # pointer must be non-null


def test_handle_accepts_cuda_stream_protocol():
    """Accepts objects implementing the __cuda_stream__ protocol."""
    proto_stream = CudaStreamProtocolObj()
    handle = DeviceResources(stream=proto_stream)
    handle.sync()


def test_handle_rejects_invalid_stream():
    """Raises TypeError for invalid stream type."""
    with pytest.raises(TypeError):
        DeviceResources(stream=1.5)
