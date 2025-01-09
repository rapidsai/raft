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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

# We always need CUDA for pylibraft because the raft dependency brings in a header-only cuco
# dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(pylibraft)

project(
  pylibraft
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# CCCL before rmm/cuco so we get the right version of CCCL
# TODO(jameslamb): is this necessary?
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
# add third party dependencies using CPM
rapids_cpm_init()
include(../../cpp/cmake/thirdparty/get_cccl.cmake)
include(../../cpp/cmake/thirdparty/get_rmm.cmake)

# TODO(jameslamb): remove this... just testing my understanding
# include(../../cpp/cmake/thirdparty/get_cutlass.cmake)

find_package(raft "${RAPIDS_VERSION}" REQUIRED COMPONENTS raft compiled)

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(pylibraft/common)
add_subdirectory(pylibraft/random)
add_subdirectory(pylibraft/sparse)
