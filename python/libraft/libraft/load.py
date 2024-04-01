# Copyright (c) 2024, NVIDIA CORPORATION.
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

import ctypes
import os


def load_library():
    # Dynamically load libraft.so. Prefer a system library if one is present to
    # avoid clobbering symbols that other packages might expect, but if no
    # other library is present use the one in the wheel.
    try:
        libraft_lib = ctypes.CDLL("libraft.so", ctypes.RTLD_GLOBAL)
    except OSError:
        libraft_lib = ctypes.CDLL(
            # TODO: Do we always know it will be lib64? Should we consider
            # finding a way for CMake to export the path for us to find here?
            os.path.join(os.path.dirname(__file__), "lib64", "libraft.so"),
            ctypes.RTLD_GLOBAL,
        )

    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where libraft was loaded from.
    return libraft_lib