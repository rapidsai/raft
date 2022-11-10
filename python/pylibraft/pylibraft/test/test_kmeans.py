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

from pylibraft.common import Handle
from pylibraft.cluster.kmeans import compute_new_centroids
from pylibraft.distance import pairwise_distance

from pylibraft.testing.utils import TestDeviceBuffer


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("additional_args", [True, False])
def test_compute_new_centroids(n_rows, n_cols, metric, n_clusters, dtype,
                               additional_args):

    order = "C"

    # A single RAFT handle can optionally be reused across
    # pylibraft functions.
    handle = Handle()

    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = TestDeviceBuffer(X, order)

    centroids = X[:n_clusters]
    centroids_device = TestDeviceBuffer(centroids, order)

    weight_per_cluster = np.zeros((n_clusters, ), dtype=dtype)
    weight_per_cluster_device = TestDeviceBuffer(weight_per_cluster, order) \
        if additional_args else None

    new_centroids = np.zeros((n_clusters, n_cols), dtype=dtype)
    new_centroids_device = TestDeviceBuffer(new_centroids, order)

    sample_weights = np.ones((n_rows,)).astype(dtype) / n_rows
    sample_weights_device = TestDeviceBuffer(sample_weights, order) \
        if additional_args else None

    # Compute new centroids naively
    dists = np.zeros((n_rows, n_clusters), dtype=dtype)
    dists_device = TestDeviceBuffer(dists, order)
    pairwise_distance(X_device, centroids_device, dists_device, metric=metric)
    handle.sync()

    labels = np.argmin(dists_device.copy_to_host(), axis=1).astype(np.int32)
    labels_device = TestDeviceBuffer(labels, order)

    expected_centers = np.empty((n_clusters, n_cols), dtype=dtype)
    expected_wX = X * sample_weights.reshape((-1, 1))
    for i in range(n_clusters):
        j = expected_wX[labels == i]
        j = j.sum(axis=0)
        g = sample_weights[labels == i].sum()
        expected_centers[i, :] = j / g

    compute_new_centroids(X_device,
                          centroids_device,
                          labels_device,
                          new_centroids_device,
                          sample_weights=sample_weights_device,
                          weight_per_cluster=weight_per_cluster_device,
                          handle=handle)

    # pylibraft functions are often asynchronous so the
    # handle needs to be explicitly synchronized
    handle.sync()

    actual_centers = new_centroids_device.copy_to_host()

    assert np.allclose(expected_centers, actual_centers, rtol=1e-6)
