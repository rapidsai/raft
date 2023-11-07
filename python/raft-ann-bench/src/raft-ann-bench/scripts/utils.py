#
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
import cupy as cp
import time
import os


def dtype_from_filename(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".fbin":
        return np.float32
    elif ext == ".ibin":
        return np.int32
    elif ext == ".u8bin":
        return np.ubyte
    elif ext == ".i8bin":
        return np.byte
    else:
        raise RuntimeError("Not supported file extension" + ext)


def suffix_from_dtype(dtype):
    if dtype == np.float32:
        return ".fbin"
    elif dtype == np.int32:
        return ".ibin"
    elif dtype == np.ubyte:
        return ".u8bin"
    elif dtype == np.byte:
        return ".i8bin"
    else:
        raise RuntimeError("Not supported dtype extension" + dtype)


def memmap_bin_file(bin_file, dtype, shape=None, mode="r", size_dtype=np.uint32):
    extent_itemsize = np.dtype(size_dtype).itemsize
    offset = int(extent_itemsize) * 2
    if bin_file is None:
        return None
    if dtype is None:
        dtype = dtype_from_filename(bin_file)

    if mode[0] == "r":
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        if shape is None:
            shape = (a[0], a[1])
            print("Read shape from file", shape)
        return np.memmap(bin_file, mode=mode, dtype=dtype, offset=offset, shape=shape)
    elif mode[0] == "w":
        if shape is None:
            raise ValueError("Need to specify shape to map file in write mode")

        print("creating file", bin_file)
        dirname = os.path.dirname(bin_file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        a[0] = shape[0]
        a[1] = shape[1]
        a.flush()
        del a
        fp = np.memmap(bin_file, mode="r+", dtype=dtype, offset=offset, shape=shape)
        return fp

    # print('# {}: shape: {}, dtype: {}'.format(bin_file, shape, dtype))


def write_bin(fname, data):
    print("writing", fname, data.shape, data.dtype, "...")
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)


def calc_recall(ann_idx, true_nn_idx):
    # ann_idx = np.asarray(ann_idx)
    ann_idx = cp.asnumpy(ann_idx)
    if ann_idx.shape != true_nn_idx.shape:
        raise RuntimeError(
            "Incompatible shapes {} vs {}".format(ann_idx.shape, true_nn_idx.shape)
        )
    n = 0
    for i in range(ann_idx.shape[0]):
        n += np.intersect1d(ann_idx[i, :], true_nn_idx[i, :]).size
    recall = n / ann_idx.size
    return recall


class BenchmarkTimer:
    """Provides a context manager that runs a code block `reps` times
    and records results to the instance variable `timings`. Use like:
    .. code-block:: python
        timer = BenchmarkTimer(rep=5)
        for _ in timer.benchmark_runs():
            ... do something ...
        print(np.min(timer.timings))

        This class is part of the rapids/cuml benchmark suite
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
