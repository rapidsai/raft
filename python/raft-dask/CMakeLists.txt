# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cpm)
include(rapids-cuda)
rapids_cuda_init_architectures(raft-dask-python)

project(
  raft-dask-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

rapids_cpm_init()
include(cmake/thirdparty/get_ucxx.cmake)

# why these components:
#
# * 'raft' = the headers, needed to link against libraft
# * 'distributed' = needed for NCCL
find_package(raft "${RAPIDS_VERSION}" REQUIRED COMPONENTS raft distributed)

include(rapids-cython-core)
rapids_cython_init()

add_subdirectory(raft_dask/common)
add_subdirectory(raft_dask/include_test)
