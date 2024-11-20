# =============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

function(find_and_configure_cutlass)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  find_package(Git REQUIRED)

  # if(RAFT_ENABLE_DIST_DEPENDENCIES OR RAFT_COMPILE_LIBRARIES)
  set(CUTLASS_ENABLE_HEADERS_ONLY
      ON
      CACHE BOOL "Enable only the header library"
  )
  set(CUTLASS_NAMESPACE
      "raft_cutlass"
      CACHE STRING "Top level namespace of CUTLASS"
  )
  set(CUTLASS_ENABLE_CUBLAS
      OFF
      CACHE BOOL "Disable CUTLASS to build with cuBLAS library."
  )

  if (CUDA_STATIC_RUNTIME)
    set(CUDART_LIBRARY "${CUDA_cudart_static_LIBRARY}" CACHE FILEPATH "fixing cutlass cmake code" FORCE)
  endif()

  rapids_cpm_find(
    NvidiaCutlass ${PKG_VERSION}
    GLOBAL_TARGETS nvidia::cutlass::cutlass
    CPM_ARGS
    GIT_REPOSITORY ${PKG_REPOSITORY}
    GIT_TAG ${PKG_PINNED_TAG}
    GIT_SHALLOW TRUE
    OPTIONS "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
    PATCH_COMMAND ${CMAKE_COMMAND} -E env GIT_COMMITTER_NAME=rapids-cmake GIT_COMMITTER_EMAIL=rapids.cmake@rapids.ai ${GIT_EXECUTABLE} am -3 ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patches/cutlass/build-export.patch
  )

  if(TARGET CUTLASS AND NOT TARGET nvidia::cutlass::cutlass)
    add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
  endif()

  if(NvidiaCutlass_ADDED)
    rapids_export(
      BUILD NvidiaCutlass
      EXPORT_SET NvidiaCutlass
      GLOBAL_TARGETS nvidia::cutlass::cutlass
      NAMESPACE nvidia::cutlass::
    )
  endif()
  # endif()

  # We generate the cutlass-config files when we built cutlass locally, so always do
  # `find_dependency`
  rapids_export_package(
          BUILD NvidiaCutlass raft-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
  )
  rapids_export_package(
          INSTALL NvidiaCutlass raft-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
  )

  # Tell cmake where it can find the generated NvidiaCutlass-config.cmake we wrote.
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
          INSTALL NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}/../]=]
          EXPORT_SET raft-exports
  )
  rapids_export_find_package_root(
          BUILD NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}]=]
          EXPORT_SET raft-exports
  )
endfunction()

set(_cutlass_version 3.5.1)
if(NOT RAFT_CUTLASS_GIT_TAG)
  set(RAFT_CUTLASS_GIT_TAG "v${_cutlass_version}")
endif()

if(NOT RAFT_CUTLASS_GIT_REPOSITORY)
  set(RAFT_CUTLASS_GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git)
endif()

find_and_configure_cutlass(
  VERSION ${_cutlass_version} REPOSITORY ${RAFT_CUTLASS_GIT_REPOSITORY} PINNED_TAG ${RAFT_CUTLASS_GIT_TAG}
)
