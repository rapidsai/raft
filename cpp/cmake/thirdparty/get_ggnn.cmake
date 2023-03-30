#=============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    set ( EXTERNAL_INCLUDES_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )
    if (NOT EXISTS ${EXTERNAL_INCLUDES_DIRECTORY}/_deps/ggnn-src/)

        execute_process (
                COMMAND git clone "https://github.com/${PKG_FORK}/ggnn" --branch ${PKG_PINNED_TAG} ggnn-src
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ )

        message("SOURCE ${CMAKE_CURRENT_SOURCE_DIR}")
        execute_process (
                COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/ggnn.patch
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps/ggnn-src
        )
    endif()

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_ggnn(VERSION          0.5
        FORK             cgtuebingen
        PINNED_TAG       release_0.5
        EXCLUDE_FROM_ALL YES)
