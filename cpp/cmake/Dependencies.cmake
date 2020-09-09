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
    GIT_REPOSITORY    https://github.com/thrust/cub.git
    GIT_TAG           1.8.0
    PREFIX            ${CUB_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")
endif(NOT CUB_IS_PART_OF_CTK)

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