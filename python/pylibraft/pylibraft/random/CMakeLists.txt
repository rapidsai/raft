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

# Set the list of Cython files to build
set(cython_sources rmat_rectangular_generator.pyx)
# TODO: should finally be replaced with 'compiled' library to be more generic,
# when that is available
set(linked_libraries raft::distance)

# Build all of the Cython targets
rapids_cython_create_modules(
    CXX
    SOURCE_FILES "${cython_sources}"
    LINKED_LIBRARIES "${linked_libraries}"
    MODULE_PREFIX random_)

foreach(cython_module IN LISTS _RAPIDS_CYTHON_CREATED_TARGETS)
    set_target_properties(${cython_module} PROPERTIES INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../")
endforeach()
