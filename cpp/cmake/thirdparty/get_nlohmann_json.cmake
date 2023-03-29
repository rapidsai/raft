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

function(find_and_configure_nlohmann_json)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(nlohmann_json ${PKG_VERSION}
            GLOBAL_TARGETS      nlohmann_json::nlohmann_json
            BUILD_EXPORT_SET    raft-bench-ann-exports
            INSTALL_EXPORT_SET  raft-bench-ann-exports
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/json.git
            GIT_TAG                ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL})

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_nlohmann_json(VERSION  3.11.2
        FORK             nlohmann
        PINNED_TAG       v3.11.2
        EXCLUDE_FROM_ALL YES)
