# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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

if(BUILD_TESTS OR BUILD_PRIMS_BENCH)
  add_library(raft_internal INTERFACE)
  target_include_directories(
    raft_internal INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/internal>"
  )
  target_compile_features(raft_internal INTERFACE cxx_std_17 $<BUILD_INTERFACE:cuda_std_17>)
endif()
