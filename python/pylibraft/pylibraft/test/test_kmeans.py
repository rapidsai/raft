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

from pylibraft.testing.utils import TestDeviceBuffer


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [100])
@pytest.mark.parametrize("n_clusters", [5])
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

    centroids = np.random.random_sample((n_clusters, n_cols)).astype(dtype)
    centroids_device = TestDeviceBuffer(centroids, order)

    l2norm_x = np.linalg.norm(X, axis=0, ord=2)
    l2norm_x_device = TestDeviceBuffer(l2norm_x, order) \
        if additional_args else None

    weight_per_cluster = np.empty((n_clusters, ), dtype=dtype)
    weight_per_cluster_device = TestDeviceBuffer(weight_per_cluster, order) \
        if additional_args else None

    new_centroids = np.empty((n_clusters, n_cols), dtype=dtype)
    new_centroids_device = TestDeviceBuffer(new_centroids, order)

    sample_weights = np.ones((n_rows,)).astype(dtype)
    sample_weights_device = TestDeviceBuffer(sample_weights, order) \
        if additional_args else None

    compute_new_centroids(X_device,
                          centroids_device,
                          new_centroids_device,
                          sample_weights=sample_weights_device,
                          l2norm_x=l2norm_x_device,
                          weight_per_cluster=weight_per_cluster_device,
                          batch_samples=n_rows,
                          batch_centroids=n_clusters,
                          handle=handle)

    # pylibraft functions are often asynchronous so the
    # handle needs to be explicitly synchronized
    handle.sync()

    print(str(new_centroids_device.copy_to_host()))

    if(additional_args):
        print(str(weight_per_cluster_device.copy_to_host()))

    # actual[actual <= 1e-5] = 0.0
    #
    # assert np.allclose(expected, actual, rtol=1e-4)
