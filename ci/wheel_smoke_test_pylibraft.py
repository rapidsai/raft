# Copyright (c) 2023, NVIDIA CORPORATION.
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
from scipy.spatial.distance import cdist

from pylibraft.common import Handle, Stream, device_ndarray
from pylibraft.distance import pairwise_distance


if __name__ == "__main__":
    metric = "euclidean"
    n_rows = 1337
    n_cols = 1337

    input1 = np.random.random_sample((n_rows, n_cols))
    input1 = np.asarray(input1, order="C").astype(np.float64)

    output = np.zeros((n_rows, n_rows), dtype=np.float64)

    expected = cdist(input1, input1, metric)

    expected[expected <= 1e-5] = 0.0

    input1_device = device_ndarray(input1)
    output_device = None

    s2 = Stream()
    handle = Handle(stream=s2)
    ret_output = pairwise_distance(
        input1_device, input1_device, output_device, metric, handle=handle
    )
    handle.sync()

    output_device = ret_output

    actual = output_device.copy_to_host()

    actual[actual <= 1e-5] = 0.0

    assert np.allclose(expected, actual, rtol=1e-4)
