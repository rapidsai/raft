#=============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.
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

include(ExternalProject)

##############################################################################
# - cub - (header only) ------------------------------------------------------

if(NOT CUB_IS_PART_OF_CTK)
  set(CUB_DIR ${CMAKE_CURRENT_BINARY_DIR}/cub CACHE STRING "Path to cub repo")
  ExternalProject_Add(cub
    GIT_REPOSITORY    https://github.com/NVIDIA/cub.git
    GIT_TAG           1.8.0
    PREFIX            ${CUB_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")
endif(NOT CUB_IS_PART_OF_CTK)

##############################################################################
# - faiss --------------------------------------------------------------------

if(BUILD_STATIC_FAISS)
  set(FAISS_DIR ${CMAKE_CURRENT_BINARY_DIR}/faiss CACHE STRING
    "Path to FAISS source directory")
  ExternalProject_Add(faiss
    GIT_REPOSITORY    https://github.com/facebookresearch/faiss.git
    GIT_TAG           a5b850dec6f1cd6c88ab467bfd5e87b0cac2e41d
    CONFIGURE_COMMAND LIBS=-pthread
                      CPPFLAGS=-w
                      LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/lib
                              ${CMAKE_CURRENT_BINARY_DIR}/faiss/src/faiss/configure
	                      --prefix=${CMAKE_CURRENT_BINARY_DIR}/faiss
	                      --with-blas=${BLAS_LIBRARIES}
	                      --with-cuda=${CUDA_TOOLKIT_ROOT_DIR}
	                      --with-cuda-arch=${FAISS_GPU_ARCHS}
	                      -v
    PREFIX            ${FAISS_DIR}
    BUILD_COMMAND     make -j${PARALLEL_LEVEL} VERBOSE=1
    BUILD_BYPRODUCTS  ${FAISS_DIR}/lib/libfaiss.a
    BUILD_ALWAYS      1
    INSTALL_COMMAND   make -s install > /dev/null
    UPDATE_COMMAND    ""
    BUILD_IN_SOURCE   1
    PATCH_COMMAND     patch -p1 -N < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/faiss_cuda11.patch || true)

  ExternalProject_Get_Property(faiss install_dir)
  add_library(FAISS::FAISS STATIC IMPORTED)
  set_property(TARGET FAISS::FAISS PROPERTY
    IMPORTED_LOCATION ${FAISS_DIR}/lib/libfaiss.a)
  # to account for the FAISS file reorg that happened recently after the current
  # pinned commit, just change the following line to
  # set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/src/faiss")
  set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/src")
else()
  add_library(FAISS::FAISS SHARED IMPORTED)
  set_property(TARGET FAISS::FAISS PROPERTY
    IMPORTED_LOCATION $ENV{CONDA_PREFIX}/lib/libfaiss.so)
  message(STATUS "Found FAISS: $ENV{CONDA_PREFIX}/lib/libfaiss.so")
endif(BUILD_STATIC_FAISS)

##############################################################################
# - googletest ---------------------------------------------------------------

if(BUILD_GTEST)

  set(GTEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/googletest CACHE STRING
    "Path to googletest repo")
  include(ExternalProject)
  ExternalProject_Add(googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           6ce9b98f541b8bcd84c5c5b3483f29a933c4aefb
    PREFIX            ${GTEST_DIR}
    CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                      -DBUILD_SHARED_LIBS=OFF
                      -DCMAKE_INSTALL_LIBDIR=lib
    BUILD_BYPRODUCTS  ${GTEST_DIR}/lib/libgtest.a
                      ${GTEST_DIR}/lib/libgtest_main.a
    UPDATE_COMMAND    "")
  add_library(GTest::GTest STATIC IMPORTED)
  add_library(GTest::Main STATIC IMPORTED)
  set_property(TARGET GTest::GTest PROPERTY
    IMPORTED_LOCATION ${GTEST_DIR}/lib/libgtest.a)
  set_property(TARGET GTest::Main PROPERTY
    IMPORTED_LOCATION ${GTEST_DIR}/lib/libgtest_main.a)
  add_dependencies(GTest::GTest googletest)
  add_dependencies(GTest::Main googletest)

else()

  find_package(GTest REQUIRED)

endif(BUILD_GTEST)

if(NOT CUB_IS_PART_OF_CTK)
  add_dependencies(GTest::GTest cub)
endif(NOT CUB_IS_PART_OF_CTK)
add_dependencies(FAISS::FAISS faiss)
