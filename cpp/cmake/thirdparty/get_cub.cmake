#=============================================================================
# Copyright (c) 2021, NVIDIA CORPORATION.
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

function(find_and_configure_cub)
    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )


    CPMFindPackage(NAME CUB
        VERSION         ${PKG_VERSION}
        GIT_REPOSITORY  https://github.com/NVlabs/cub.git
        GIT_TAG         ${PKG_PINNED_TAG}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   TRUE
    )

    set(CUB_SOURCE_DIR "${CUB_SOURCE_DIR}" PARENT_SCOPE)
endfunction()

find_and_configure_cub(VERSION     1.10.0
                       PINNED_TAG  1.10.0)
