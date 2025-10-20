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
  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

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

  include("${rapids-cmake-dir}/cpm/package_override.cmake")
  rapids_cpm_package_override("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches/cutlass_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(cutlass VERSION_VAR version FIND_VAR find_args CPM_VAR cpm_find_info
                          TO_INSTALL_VAR to_install)

  rapids_cpm_find(
    NvidiaCutlass ${version} ${find_args}
    GLOBAL_TARGETS nvidia::cutlass::cutlass
    CPM_ARGS ${cpm_find_info}
    OPTIONS "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
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

find_and_configure_cutlass()
