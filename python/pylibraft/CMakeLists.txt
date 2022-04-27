# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

set(raft_version 22.06.00)

set(CYTHON_FLAGS
    "--directive binding=True,embedsignature=True,always_allow_keywords=True"
    CACHE STRING "The directives for Cython compilation.")

project(
  raft-python
  VERSION ${raft_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C
            CXX
            # TODO: The C++ RMM CMake configuration targets cuda_std_17 features, which prior to
            # CMake 3.22 will also pull in the corresponding required languages even if this project
            # does not actually require those languages. As a result, we need to include CUDA here.
            # We can remove CUDA once we upgrade the minimum required CMake version to 3.22.
            CUDA)

option(FIND_RAFT_CPP "Search for existing RAFT C++ installations before defaulting to local files"
       OFF)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

# Ignore unused variable warning.
set(ignored_variable "${SKBUILD}")

# If the user requested it we attempt to find RAFT.
if(FIND_RAFT_CPP)
  find_package(RAFT ${raft_version})
else()
  set(RAFT_FOUND OFF)
endif()

if(NOT RAFT_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  set(BUILD_TESTS OFF)
  set(RAFT_COMPILE_LIBRARIES ON)
  set(RAFT_COMPILE_DIST_LIBRARY ON)
  #RAFT_COMPILE_NN_LIBRARY  OFF
  #RAFT_ENABLE_NN_DEPENDENCIES  ON
  add_subdirectory(../../cpp raft-cpp)
endif()

add_subdirectory(cmake)
add_subdirectory(pylibraft/distance)
