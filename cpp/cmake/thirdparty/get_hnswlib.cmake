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

function(find_and_configure_hnswlib)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    set ( EXTERNAL_INCLUDES_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
    if( NOT EXISTS ${EXTERNAL_INCLUDES_DIRECTORY}/_deps/hnswlib-src )

        execute_process (
                COMMAND git clone --branch=v0.6.2 https://github.com/nmslib/hnswlib.git hnswlib-src
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps )

    endif ()

    include(cmake/modules/FindAVX.cmake)

    set(HNSW_CXX_FLAGS "")
    if(CXX_AVX_FOUND)
        set(HNSW_CXX_FLAGS "${HNSW_CXX_FLAGS} ${CXX_AVX_FLAGS}")
    elseif(CXX_AVX2_FOUND)
        set(HNSW_CXX_FLAGS "${HNSW_CXX_FLAGS} ${CXX_AVX2_FLAGS}")
    elseif(CXX_AVX512_FOUND)
        set(HNSW_CXX_FLAGS "${HNSW_CXX_FLAGS} ${CXX_AVX512_FLAGS}")
    endif()
endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_hnswlib(VERSION  0.6.2
        FORK             nmslib
        PINNED_TAG       v0.6.2
        EXCLUDE_FROM_ALL YES)
