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

function(find_and_configure_nccl)

    if(TARGET nccl::nccl)
        return()
    endif()

    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    GENERATE_FIND_MODULE(
        NAME         nccl
        HEADER_NAME  nccl.h
        LIBRARY_NAME nccl
    )

    CPMFindPackage(NAME nccl
        VERSION         ${PKG_VERSION}
        GIT_REPOSITORY  https://github.com/NVIDIA/nccl.git
        GIT_TAG         ${PKG_PINNED_TAG}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   YES
    )

    set(NCCL_SOURCE_DIR "${RAFT_SOURCE_DIR}" PARENT_SCOPE)

    if (nccl_ADDED)
        # todo (DD): Add building nccl from source, works fine for conda installed for now


    endif()
endfunction()

find_and_configure_nccl(VERSION     "${RAFT_VERSION_MAJOR}.${RAFT_VERSION_MINOR}"
                        PINNED_TAG  911d61f214d45c98df1ee8c0ac23c33fb94b63de)




