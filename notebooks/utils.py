#
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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


import cupy as cp
import h5py
import os
import tempfile
import time
import urllib

## Check the quality of the prediction (recall)
def calc_recall(found_indices, ground_truth):
    found_indices = cp.asarray(found_indices)
    bs, k = found_indices.shape
    if bs != ground_truth.shape[0]:
        raise RuntimeError(
            "Batch sizes do not match {} vs {}".format(
                bs, ground_truth.shape[0]
            )
        )
    if k > ground_truth.shape[1]:
        raise RuntimeError(
            "Not enough indices in the ground truth ({} > {})".format(
                k, ground_truth.shape[1]
            )
        )
    n = 0
    # Go over the batch
    for i in range(bs):
        # Note, ivf-pq does not guarantee the ordered input, hence the use of intersect1d
        n += cp.intersect1d(found_indices[i, :k], ground_truth[i, :k]).size
    recall = n / found_indices.size
    return recall


class BenchmarkTimer:
    """Provides a context manager that runs a code block `reps` times
    and records results to the instance variable `timings`. Use like:
    .. code-block:: python
        timer = BenchmarkTimer(rep=5)
        for _ in timer.benchmark_runs():
            ... do something ...
        print(np.min(timer.timings))

        This class is borrowed from the rapids/cuml benchmark suite
    """

    def __init__(self, reps=1, warmup=0):
        self.warmup = warmup
        self.reps = reps
        self.timings = []

    def benchmark_runs(self):
        for r in range(self.reps + self.warmup):
            t0 = time.time()
            yield r
            t1 = time.time()
            self.timings.append(t1 - t0)
            if r >= self.warmup:
                self.timings.append(t1 - t0)


def load_dataset(dataset_url="http://ann-benchmarks.com/sift-128-euclidean.hdf5", work_folder=None):
    """Download dataset from url. It is expected that the dataset contains a hdf5 file in ann-benchmarks format

    Parameters
    ----------
      dataset_url address of hdf5 file
      work_folder name of the local folder to store the dataset

    """
    dataset_filename = dataset_url.split("/")[-1]

    # We'll need to load store some data in this tutorial
    if work_folder is None:
        work_folder = os.path.join(tempfile.gettempdir(), "raft_example")

    if not os.path.exists(work_folder):
        os.makedirs(work_folder)
    print("The index and data will be saved in", work_folder)

    ## download the dataset
    dataset_path = os.path.join(work_folder, dataset_filename)
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve(dataset_url, dataset_path)

    f = h5py.File(dataset_path, "r")

    return f
