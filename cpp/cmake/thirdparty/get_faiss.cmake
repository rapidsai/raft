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

function(find_and_configure_faiss)
    set(oneValueArgs VERSION PINNED_TAG)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )

    GENERATE_FIND_MODULE(
        NAME         FAISS
        HEADER_NAME  faiss/IndexFlat.h
        LIBRARY_NAME faiss
    )

    CPMFindPackage(NAME FAISS
        VERSION         ${PKG_VERSION}
        GIT_REPOSITORY  https://github.com/facebookresearch/faiss.git
        GIT_TAG         ${PKG_PINNED_TAG}
        OPTIONS
          "FAISS_ENABLE_PYTHON OFF"
          "BUILD_SHARED_LIBS OFF"
          "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
          "FAISS_ENABLE_GPU ON"
          "BUILD_TESTING OFF"
    )

    set(FAISS_LIBRARIES ${FAISS_LIBRARY_RELEASE} PARENT_SCOPE)

    # todo (DD): Test building faiss statically and add gpu archs again
endfunction()

find_and_configure_faiss(VERSION    1.7.0
                        PINNED_TAG  7c2d2388a492d65fdda934c7e74ae87acaeed066
                        )

