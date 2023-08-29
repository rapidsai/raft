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


import sys

import h5py
import numpy as np


def normalize(x):
    norm = np.linalg.norm(x, axis=1)
    return (x.T / norm).T


def write_bin(fname, data):
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)


if __name__ == "__main__":
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print(
            "usage: %s [-n] <input>.hdf5\n" % (sys.argv[0]),
            "  -n: normalize base/query set\n",
            "outputs: <input>.base.fbin\n",
            "         <input>.query.fbin\n",
            "         <input>.groundtruth.neighbors.ibin\n",
            "         <input>.groundtruth.distances.fbin",
            file=sys.stderr,
        )
        sys.exit(-1)

    need_normalize = False
    if len(sys.argv) == 3:
        assert sys.argv[1] == "-n"
        need_normalize = True
    fname_prefix = sys.argv[-1]
    assert fname_prefix.endswith(".hdf5")
    fname_prefix = fname_prefix[:-5]

    hdf5 = h5py.File(sys.argv[-1], "r")
    assert (
        hdf5.attrs["distance"] == "angular"
        or hdf5.attrs["distance"] == "euclidean"
    )
    assert hdf5["train"].dtype == np.float32
    assert hdf5["test"].dtype == np.float32
    assert hdf5["neighbors"].dtype == np.int32
    assert hdf5["distances"].dtype == np.float32

    base = hdf5["train"][:]
    query = hdf5["test"][:]
    if need_normalize:
        base = normalize(base)
        query = normalize(query)
    elif hdf5.attrs["distance"] == "angular":
        print(
            "warning: input has angular distance, ",
            "specify -n to normalize base/query set!\n",
        )

    output_fname = fname_prefix + ".base.fbin"
    print("writing", output_fname, "...")
    write_bin(output_fname, base)

    output_fname = fname_prefix + ".query.fbin"
    print("writing", output_fname, "...")
    write_bin(output_fname, query)

    output_fname = fname_prefix + ".groundtruth.neighbors.ibin"
    print("writing", output_fname, "...")
    write_bin(output_fname, hdf5["neighbors"][:])

    output_fname = fname_prefix + ".groundtruth.distances.fbin"
    print("writing", output_fname, "...")
    write_bin(output_fname, hdf5["distances"][:])
