#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

function(find_and_configure_ggnn)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
          "${multiValueArgs}" ${ARGN} )


  set(patch_files_to_run "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/ggnn.diff")
  set(patch_issues_to_ref "fix compile issues")
  set(patch_script "${CMAKE_BINARY_DIR}/rapids-cmake/patches/ggnn/patch.cmake")
  set(log_file "${CMAKE_BINARY_DIR}/rapids-cmake/patches/ggnn/log")
  string(TIMESTAMP current_year "%Y" UTC)
  configure_file(${rapids-cmake-dir}/cpm/patches/command_template.cmake.in "${patch_script}"
                @ONLY)

  rapids_cpm_find(
    ggnn ${PKG_VERSION}
    GLOBAL_TARGETS ggnn::ggnn
    CPM_ARGS
    GIT_REPOSITORY ${PKG_REPOSITORY}
    GIT_TAG ${PKG_PINNED_TAG}
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY ON
    PATCH_COMMAND ${CMAKE_COMMAND} -P ${patch_script}
  )
  if(NOT TARGET ggnn::ggnn)
    add_library(ggnn INTERFACE)
    target_include_directories(ggnn INTERFACE "$<BUILD_INTERFACE:${ggnn_SOURCE_DIR}/include>")
    add_library(ggnn::ggnn ALIAS ggnn)
  endif()

endfunction()
if(NOT RAFT_GGNN_GIT_TAG)
  set(RAFT_GGNN_GIT_TAG release_0.5)
endif()

if(NOT RAFT_GGNN_GIT_REPOSITORY)
  set(RAFT_GGNN_GIT_REPOSITORY https://github.com/cgtuebingen/ggnn.git)
endif()
find_and_configure_ggnn(VERSION 0.5
        REPOSITORY       ${RAFT_GGNN_GIT_REPOSITORY}
        PINNED_TAG       ${RAFT_GGNN_GIT_TAG}
        )
