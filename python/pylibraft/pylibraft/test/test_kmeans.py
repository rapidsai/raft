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

from pylibraft.cluster.kmeans import (
    KMeansParams,
    cluster_cost,
    compute_new_centroids,
    fit,
    init_plus_plus,
)
from pylibraft.common import DeviceResources, device_ndarray
from pylibraft.distance import pairwise_distance


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_fit(n_rows, n_cols, n_clusters, dtype):
    # generate some random input points / centroids
    X_host = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    centroids = device_ndarray(X_host[:n_clusters])
    X = device_ndarray(X_host)

    # compute the inertia, before fitting centroids
    original_inertia = cluster_cost(X, centroids)

    params = KMeansParams(n_clusters=n_clusters, seed=42)

    # fit the centroids, make sure inertia has gone down
    # TODO: once we have make_blobs exposed to python
    # (https://github.com/rapidsai/raft/issues/1059)
    # we should use that to test out the kmeans fit, like the C++
    # tests do right now
    centroids, inertia, n_iter = fit(params, X, centroids)
    assert inertia < original_inertia
    assert n_iter >= 1
    assert np.allclose(cluster_cost(X, centroids), inertia, rtol=1e-6)


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [5, 15])
@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("additional_args", [True, False])
def test_compute_new_centroids(
    n_rows, n_cols, metric, n_clusters, dtype, additional_args
):

    # A single RAFT handle can optionally be reused across
    # pylibraft functions.
    handle = DeviceResources()

    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = X[:n_clusters]
    centroids_device = device_ndarray(centroids)

    weight_per_cluster = np.zeros((n_clusters,), dtype=dtype)
    weight_per_cluster_device = (
        device_ndarray(weight_per_cluster) if additional_args else None
    )

    new_centroids = np.zeros((n_clusters, n_cols), dtype=dtype)
    new_centroids_device = device_ndarray(new_centroids)

    sample_weights = np.ones((n_rows,)).astype(dtype) / n_rows
    sample_weights_device = (
        device_ndarray(sample_weights) if additional_args else None
    )

    # Compute new centroids naively
    dists = np.zeros((n_rows, n_clusters), dtype=dtype)
    dists_device = device_ndarray(dists)
    pairwise_distance(X_device, centroids_device, dists_device, metric=metric)
    handle.sync()

    labels = np.argmin(dists_device.copy_to_host(), axis=1).astype(np.int32)
    labels_device = device_ndarray(labels)

    expected_centers = np.empty((n_clusters, n_cols), dtype=dtype)
    expected_wX = X * sample_weights.reshape((-1, 1))
    for i in range(n_clusters):
        j = expected_wX[labels == i]
        j = j.sum(axis=0)
        g = sample_weights[labels == i].sum()
        expected_centers[i, :] = j / g

    compute_new_centroids(
        X_device,
        centroids_device,
        labels_device,
        new_centroids_device,
        sample_weights=sample_weights_device,
        weight_per_cluster=weight_per_cluster_device,
        handle=handle,
    )

    # pylibraft functions are often asynchronous so the
    # handle needs to be explicitly synchronized
    handle.sync()

    actual_centers = new_centroids_device.copy_to_host()

    assert np.allclose(expected_centers, actual_centers, rtol=1e-6)


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [4, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cluster_cost(n_rows, n_cols, n_clusters, dtype):
    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = X[:n_clusters]
    centroids_device = device_ndarray(centroids)

    inertia = cluster_cost(X_device, centroids_device)

    # compute the nearest centroid to each sample
    distances = pairwise_distance(
        X_device, centroids_device, metric="sqeuclidean"
    ).copy_to_host()
    cluster_ids = np.argmin(distances, axis=1)

    cluster_distances = np.take_along_axis(
        distances, cluster_ids[:, None], axis=1
    )

    # need reduced tolerance for float32
    tol = 1e-3 if dtype == np.float32 else 1e-6
    assert np.allclose(inertia, sum(cluster_distances), rtol=tol, atol=tol)


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [4, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_init_plus_plus(n_rows, n_cols, n_clusters, dtype):
    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = init_plus_plus(X_device, n_clusters, seed=1)
    centroids_ = centroids.copy_to_host()

    assert centroids_.shape == (n_clusters, X.shape[1])

    # Centroids are selected from the existing points
    for centroid in centroids_:
        assert (centroid == X).all(axis=1).any()


@pytest.mark.parametrize("n_rows", [100])
@pytest.mark.parametrize("n_cols", [5, 25])
@pytest.mark.parametrize("n_clusters", [4, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_init_plus_plus_preallocated_output(n_rows, n_cols, n_clusters, dtype):
    X = np.random.random_sample((n_rows, n_cols)).astype(dtype)
    X_device = device_ndarray(X)

    centroids = device_ndarray.empty((n_clusters, n_cols), dtype=dtype)

    new_centroids = init_plus_plus(X_device, centroids=centroids, seed=1)
    new_centroids_ = new_centroids.copy_to_host()

    # The shape should not have changed
    assert new_centroids_.shape == centroids.shape

    # Centroids are selected from the existing points
    for centroid in new_centroids_:
        assert (centroid == X).all(axis=1).any()


def test_init_plus_plus_exclusive_arguments():
    # Check an exception is raised when n_clusters and centroids shape
    # are inconsistent.
    X = np.random.random_sample((10, 5)).astype(np.float64)
    X = device_ndarray(X)

    n_clusters = 3

    centroids = np.random.random_sample((n_clusters + 1, 5)).astype(np.float64)
    centroids = device_ndarray(centroids)

    with pytest.raises(
        RuntimeError, match="Parameters 'n_clusters' and 'centroids'"
    ):
        init_plus_plus(X, n_clusters, centroids=centroids)
