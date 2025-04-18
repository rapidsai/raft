# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

option(USE_NCCL_RUNTIME_WHEEL "Use the NCCL wheel at runtime instead of the system library" OFF)

set(cython_sources comms_utils.pyx nccl.pyx)
set(linked_libraries raft::raft raft::distributed)
rapids_cython_create_modules(
  SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" CXX
)

if(USE_NCCL_RUNTIME_WHEEL)
  set(rpaths "$ORIGIN/../../nvidia/nccl/lib")
  foreach(tgt IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
    set_property(
      TARGET ${tgt}
      PROPERTY INSTALL_RPATH ${rpaths}
      APPEND
    )
  endforeach()
endif()
