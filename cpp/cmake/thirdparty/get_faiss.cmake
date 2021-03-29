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
        NAME         faiss
        HEADER_NAME  faiss.h
        LIBRARY_NAME faiss
    )

    CPMFindPackage(NAME faiss
        VERSION         ${PKG_VERSION}
        GIT_REPOSITORY  https://github.com/facebookresearch/faiss.git
        GIT_TAG         ${PKG_PINNED_TAG}
        GIT_SHALLOW     TRUE
        DOWNLOAD_ONLY   YES
        # OPTIONS         "CMAKE_BUILD_TYPE Release
        #                 BUILD_TESTING OFF
        #                 FAISS_ENABLE_PYTHON OFF
        #                 BUILD_SHARED_LIBS OFF
        #                 FAISS_ENABLE_GPU ON
        #                 CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}
        #                 CUDA_ARCHITECTURES ${FAISS_GPU_ARCHS}
        #                 BLAS_LIBRARIES ${BLAS_LIBRARIES}"
    )

    set(faiss_SOURCE_DIR "${faiss_SOURCE_DIR}" PARENT_SCOPE)

    if (faiss_ADDED)
        # todo (DD): Test building faiss statically


    endif()
endfunction()

find_and_configure_faiss(VERSION    0
                        PINNED_TAG  c65f6705236570c9aac365c68e3cc5da2343c888
                        )

