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

set(cython_sources comms_utils.pyx nccl.pyx)
set(linked_libraries raft::raft)
rapids_cython_create_modules(
    SOURCE_FILES "${cython_sources}"
    LINKED_LIBRARIES "${linked_libraries}"
    CXX)

find_library(NCCL_PATH nccl REQUIRED)
foreach(cython_module IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
  set(is_nccl "")
  string(FIND "${cython_module}" nccl is_nccl)

  if(NOT is_nccl EQUAL -1)
    target_link_libraries("${cython_module}" "${NCCL_PATH}")
  endif()
endforeach()
