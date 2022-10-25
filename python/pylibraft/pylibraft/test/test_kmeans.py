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

from pylibraft.common import Handle
from pylibraft.cluster.kmeans import compute_new_centroids
from pylibraft.distance import fused_l2_nn_argmin

from pylibraft.testing.utils import TestDeviceBuffer


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [100])
@pytest.mark.parametrize("n_clusters", [5])
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_new_centroids(n_rows, n_cols, metric, n_clusters, dtype):

    order = "C"

    # A single RAFT handle can optionally be reused across
    # pylibraft functions.
    handle = Handle()

    X = np.random.random_sample((n_rows, n_cols)).astype(np.float32)
    centroids = np.random.random_sample((n_clusters, n_cols)).astype(np.float32)


    l2norm_x = np.linalg.norm(X, axis=0, ord=2)



    new_weight = np.empty((n_clusters, ), dtype=np.float32)
    new_centroids = np.empty((n_clusters, n_cols), dtype=np.float32)

    X_device = TestDeviceBuffer(X, order)
    centroids_device = TestDeviceBuffer(centroids, order)

    argmin = np.empty((n_rows, ), dtype=np.int32)
    argmin_device = TestDeviceBuffer(argmin, order)

    weight, _ = np.histogram(argmin_device.copy_to_host(), bins=np.arange(0, n_clusters+1))
    weight = weight.astype(np.float32)

    weight_device = TestDeviceBuffer(weight, order)

    fused_l2_nn_argmin(centroids_device, X_device, argmin_device, handle=handle)

    new_weight_device = TestDeviceBuffer(new_weight, order)
    new_centroids_device = TestDeviceBuffer(new_centroids, order)
    l2norm_x_device = TestDeviceBuffer(l2norm_x, order)

    compute_new_centroids(X_device,
                     centroids_device,
                     weight_device,
                     l2norm_x_device,
                     new_centroids_device,
                     new_weight_device,
                     n_rows,
                     n_clusters)

    # pylibraft functions are often asynchronous so the
    # handle needs to be explicitly synchronized
    handle.sync()

    print(str(new_centroids))
    print(str(new_weight))

    # actual[actual <= 1e-5] = 0.0
    #
    # assert np.allclose(expected, actual, rtol=1e-4)
