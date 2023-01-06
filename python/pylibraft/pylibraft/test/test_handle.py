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

from pylibraft.common import Handle, Stream, device_ndarray
from pylibraft.distance import pairwise_distance

try:
    import cupy
except ImportError:
    pytest.skip(reason="cupy not installed.")


@pytest.mark.parametrize("stream", [cupy.cuda.Stream().ptr, Stream()])
def test_auto_convert_output(stream):

    input1 = np.random.random_sample((50, 3))
    input1 = np.asarray(input1, order="F").astype("float")

    output = np.zeros((50, 50), dtype="float")

    input1_device = device_ndarray(input1)
    output_device = device_ndarray(output)

    # We are just testing that this doesn't segfault
    handle = Handle(stream)
    pairwise_distance(
        input1_device, input1_device, output_device, "euclidean", handle=handle
    )
    handle.sync()
