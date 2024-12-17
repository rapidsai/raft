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
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(raft-dask-python)

project(
  raft-dask-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

rapids_cpm_init()
# TODO(jameslamb): could this just use the libucxx wheel?
# Once https://github.com/rapidsai/ucxx/issues/173 is resolved we can remove this.
find_package(ucx REQUIRED)
include(cmake/thirdparty/get_ucxx.cmake)

# raft-dask doesn't actually use raft libraries, it just needs the headers
find_package(raft "${RAPIDS_VERSION}" REQUIRED COMPONENTS raft)

# list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/cmake/find_modules)
# find_package(NCCL REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(raft_dask/common)
add_subdirectory(raft_dask/include_test)
