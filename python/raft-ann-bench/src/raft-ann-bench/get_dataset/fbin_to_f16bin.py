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


from __future__ import absolute_import, division, print_function

import sys

import numpy as np


def read_fbin(fname):
    shape = np.fromfile(fname, dtype=np.uint32, count=2)
    if float(shape[0]) * shape[1] * 4 > 2000000000:
        data = np.memmap(fname, dtype=np.float32, offset=8, mode="r").reshape(
            shape
        )
    else:
        data = np.fromfile(fname, dtype=np.float32, offset=8).reshape(shape)
    return data


def write_bin(fname, data):
    with open(fname, "wb") as f:
        np.asarray(data.shape, dtype=np.uint32).tofile(f)
        data.tofile(f)


if len(sys.argv) != 3:
    print(
        "usage: %s input.fbin output.f16bin" % (sys.argv[0]),
        file=sys.stderr,
    )
    sys.exit(-1)

data = read_fbin(sys.argv[1]).astype(np.float16)
write_bin(sys.argv[2], data)
