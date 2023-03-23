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
from scipy.spatial.distance import cdist

from pylibraft.common import DeviceResources, Stream, device_ndarray
from pylibraft.neighbors.brute_force import knn


@pytest.mark.parametrize("n_index_rows", [32, 100])
@pytest.mark.parametrize("n_query_rows", [32, 100])
@pytest.mark.parametrize("n_cols", [40, 100])
@pytest.mark.parametrize("k", [1, 5, 32])
@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
        "cityblock",
        "chebyshev",
        "canberra",
        "correlation",
        "russellrao",
        "cosine",
        "sqeuclidean",
        # "inner_product",
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32])
def test_knn(
    n_index_rows, n_query_rows, n_cols, k, inplace, metric, order, dtype
):
    index = np.random.random_sample((n_index_rows, n_cols)).astype(dtype)
    queries = np.random.random_sample((n_query_rows, n_cols)).astype(dtype)

    # RussellRao expects boolean arrays
    if metric == "russellrao":
        index[index < 0.5] = 0.0
        index[index >= 0.5] = 1.0
        queries[queries < 0.5] = 0.0
        queries[queries >= 0.5] = 1.0

    indices = np.zeros((n_query_rows, k), dtype="int64")
    distances = np.zeros((n_query_rows, k), dtype=dtype)

    index_device = device_ndarray(index)

    queries_device = device_ndarray(queries)
    indices_device = device_ndarray(indices)
    distances_device = device_ndarray(distances)

    s2 = Stream()
    handle = DeviceResources(stream=s2)
    ret_distances, ret_indices = knn(
        index_device,
        queries_device,
        k,
        indices=indices_device,
        distances=distances_device,
        metric=metric,
        handle=handle,
    )
    handle.sync()

    pw_dists = cdist(queries, index, metric=metric)

    distances_device = ret_distances if not inplace else distances_device

    actual_distances = distances_device.copy_to_host()

    actual_distances[actual_distances <= 1e-5] = 0.0
    argsort = np.argsort(pw_dists, axis=1)

    for i in range(pw_dists.shape[0]):
        expected_indices = argsort[i]
        gpu_dists = actual_distances[i]

        if metric == "correlation" or metric == "cosine":
            gpu_dists = gpu_dists[::-1]

        cpu_ordered = pw_dists[i, expected_indices]
        np.testing.assert_allclose(
            cpu_ordered[:k], gpu_dists, atol=1e-4, rtol=1e-4
        )
