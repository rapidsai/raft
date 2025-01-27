# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

# If libucx was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libucx
except ModuleNotFoundError:
    pass
else:
    libucx.load_library()
    del libucx

# If libraft was installed as a wheel, we must request it to load the library
# symbols. Otherwise, we assume that the library was installed in a system path that ld
# can find.
try:
    import libraft
except ModuleNotFoundError:
    pass
else:
    libraft.load_library()
    del libraft

from raft_dask._version import __git_commit__, __version__
