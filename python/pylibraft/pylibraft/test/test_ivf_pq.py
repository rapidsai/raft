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
from sklearn.neighbors import NearestNeighbors

from pylibraft.neighbors import IvfPq

from pylibraft.testing.utils import TestDeviceBuffer


def generate_dataset(shape, dtype):
    if dtype in [np.float32]:
        x = np.random.random_sample(shape).astype(dtype)
    elif dtype == np.byte:
        x = np.random.randint(-127, size=shape, dtype=np.byte)
    elif dtype == np.ubyte:
        x = np.random.randint(0, size=shape, dtype=np.ubyte)
    return x


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [100])
@pytest.mark.parametrize("n_queries", [20])
@pytest.mark.parametrize("k", [0])
@pytest.mark.parametrize("n_lists", [10])
@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
    ],
)
@pytest.mark.parametrize("dtype", [np.float32])
def test_ivf_pq_build(n_rows, n_cols, n_queries, k, n_lists, metric, dtype):
    dataset = generate_data((n_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)

    out_idx = np.zeros((n_queries, k), dtype=dtype)
    out_dist = np.zeros((n_queries, k), dtype=dtype)

    dataset_device = TestDeviceBuffer(dataset, order="C")
    queries_device = TestDeviceBuffer(queries, order="C")

    nn = IvfPq()

    nn.build(
        dataset_device,
        n_lists,
        metric,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=0.5,
        pq_bits=8,
        pq_dim=0,
        codebok_kind="per_subspace",
        force_random_rotation=False,
        add_data_on_build=True,
    )

    # actual = output_device.copy_to_host()

    # actual[actual <= 1e-5] = 0.0

    assert nn._index is not None
    # assert np.allclose(expected, actual, rtol=1e-4)
