#=============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

function(find_and_configure_glog)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(glog ${PKG_VERSION}
            GLOBAL_TARGETS      glog::glog
            BUILD_EXPORT_SET    raft-exports
            INSTALL_EXPORT_SET  raft-exports
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/glog.git
            GIT_TAG                ${PKG_PINNED_TAG}
            SOURCE_SUBDIR          cpp
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(glog_ADDED)
        message(VERBOSE "RAFT: Using glog located in ${glog_SOURCE_DIR}")
    else()
        message(VERBOSE "RAFT: Using glog located in ${glog_DIR}")
    endif()


endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_glog_SOURCE=/path/to/local/glog
find_and_configure_glog(VERSION 0.6.0
        FORK             google
        PINNED_TAG       v0.6.0
        EXCLUDE_FROM_ALL ON
        )