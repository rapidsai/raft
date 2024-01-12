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

function(find_and_configure_hnswlib)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG EXCLUDE_FROM_ALL)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
          "${multiValueArgs}" ${ARGN} )

  set(patch_files_to_run "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/hnswlib.diff")
  set(patch_issues_to_ref "fix compile issues")
  set(patch_script "${CMAKE_BINARY_DIR}/rapids-cmake/patches/hnswlib/patch.cmake")
  set(log_file "${CMAKE_BINARY_DIR}/rapids-cmake/patches/hnswlib/log")
  string(TIMESTAMP current_year "%Y" UTC)
  configure_file(${rapids-cmake-dir}/cpm/patches/command_template.cmake.in "${patch_script}"
                @ONLY)

  rapids_cpm_find(
    hnswlib ${PKG_VERSION}
    GLOBAL_TARGETS hnswlib::hnswlib
    CPM_ARGS
    GIT_REPOSITORY ${PKG_REPOSITORY}
    GIT_TAG ${PKG_PINNED_TAG}
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL  ${PKG_EXCLUDE_FROM_ALL}
    PATCH_COMMAND ${CMAKE_COMMAND} -P ${patch_script}
  )
  if(NOT TARGET hnswlib::hnswlib)
    add_library(hnswlib::hnswlib ALIAS hnswlib)
  endif()
endfunction()


if(NOT RAFT_HNSWLIB_GIT_TAG)
  set(RAFT_HNSWLIB_GIT_TAG v0.6.2)
endif()

if(NOT RAFT_HNSWLIB_GIT_REPOSITORY)
  set(RAFT_HNSWLIB_GIT_REPOSITORY https://github.com/nmslib/hnswlib.git)
endif()
find_and_configure_hnswlib(VERSION 0.6.2
        REPOSITORY       ${RAFT_HNSWLIB_GIT_REPOSITORY}
        PINNED_TAG       ${RAFT_HNSWLIB_GIT_TAG}
        EXCLUDE_FROM_ALL YES)
