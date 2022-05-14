# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# TODO: This is a pretty hacky way to handle the NCCL dependency. There are a
# couple of reasons I'm doing it this way right now:
# 1. The nccl target name ends up making it impossible to specify that as a target library.
# 2. nccl doesn't ship with a config file or FindNCCL.cmake file.
find_library(NCCL_PATH nccl REQUIRED)

set(cython_sources comms_utils.pyx nccl.pyx)
set(linked_libraries raft::raft "${NCCL_PATH}")
rapids_cython_create_modules(
    SOURCE_FILES "${cython_sources}"
    LINKED_LIBRARIES "${linked_libraries}"
    CXX)
