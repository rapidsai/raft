#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#=============================================================================

function(find_and_configure_cutlass)
    set(oneValueArgs VERSION REPOSITORY PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    if(RAFT_ENABLE_DIST_DEPENDENCIES OR RAFT_COMPILE_LIBRARIES)
      rapids_find_generate_module(cutlass
          HEADER_NAMES  cutlass/include/*
          INCLUDE_SUFFIXES cutlass
      )
      set(CUTLASS_ENABLE_HEADERS_ONLY ON)
      set(RAFT_CUTLASS_NAMESPACE "raft_cutlass" CACHE STRING "Top level namespace of CUTLASS")
#      set(BUILD_SHARED_LIBS OFF)
      # if (PKG_BUILD_STATIC_LIBS)
      #   set(BUILD_SHARED_LIBS OFF)
      #   set(CPM_DOWNLOAD_cutlass ON)
      # endif()

      rapids_cpm_find(cutlass ${PKG_VERSION}
          GLOBAL_TARGETS     cutlass::cutlass
          CPM_ARGS
            GIT_REPOSITORY   ${PKG_REPOSITORY}
            GIT_TAG          ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL ${PKG_EXCLUDE_FROM_ALL}
            OPTIONS
              "CMAKE_INSTALL_INCLUDEDIR include"
              "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
      )

      if(TARGET cutlass AND NOT TARGET cutlass::cutlass)
          add_library(cutlass::cutlass ALIAS cutlass)
      endif()

      # if(cutlass_ADDED)
      #   rapids_export(BUILD cutlass
      #       EXPORT_SET cutlass-targets
      #       GLOBAL_TARGETS cutlass
      #       NAMESPACE cutlass::)
      # endif()
    endif()

    # We generate the faiss-config files when we built faiss locally, so always do `find_dependency`
    rapids_export_package(BUILD cutlass raft-distance-lib-exports GLOBAL_TARGETS cutlass::cutlass cutlass)
    rapids_export_package(INSTALL cutlass raft-distance-lib-exports GLOBAL_TARGETS cutlass::cutlass cutlass)

    # Tell cmake where it can find the generated cutlass-config.cmake we wrote.
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(INSTALL cutlass [=[${CMAKE_CURRENT_LIST_DIR}]=] raft-distance-lib-exports)
endfunction()

if(NOT RAFT_CUTLASS_GIT_TAG)
  set(RAFT_CUTLASS_GIT_TAG v2.9.0)
endif()

if(NOT RAFT_CUTLASS_GIT_REPOSITORY)
  set(RAFT_CUTLASS_GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git)
endif()

find_and_configure_cutlass(VERSION    2.9.0
                         REPOSITORY  ${RAFT_CUTLASS_GIT_REPOSITORY}
                         PINNED_TAG  ${RAFT_CUTLASS_GIT_TAG})
