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
import numpy as np
import pytest

try:
    import cupy
except ImportError:
    pytest.skip(reason="cupy not installed.")

import pylibraft.config
from pylibraft.common import auto_convert_output, device_ndarray


@auto_convert_output
def gen_cai(m, n, t=None):
    if t is None:
        return device_ndarray.empty((m, n))
    elif t == tuple:
        return device_ndarray.empty((m, n)), device_ndarray.empty((m, n))
    elif t == list:
        return [device_ndarray.empty((m, n)), device_ndarray.empty((m, n))]


@pytest.mark.parametrize(
    "out_type",
    [
        ["cupy", cupy.ndarray],
        ["raft", pylibraft.common.device_ndarray],
        [lambda arr: arr.copy_to_host(), np.ndarray],
    ],
)
@pytest.mark.parametrize("gen_t", [None, tuple, list])
def test_auto_convert_output(out_type, gen_t):

    conf, t = out_type
    pylibraft.config.set_output_as(conf)

    output = gen_cai(1, 5, gen_t)

    if not isinstance(output, (list, tuple)):
        assert isinstance(output, t)

    else:
        for o in output:
            assert isinstance(o, t)

    # Make sure we set the config back to default
    pylibraft.config.set_output_as("raft")
