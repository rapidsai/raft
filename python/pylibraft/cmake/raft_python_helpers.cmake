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

#[=======================================================================[.rst:
add_cython_modules
------------------

Generate C(++) from Cython and create Python modules.

.. code-block:: cmake

  add_cython_modules(<ModuleName...>)

Creates a Cython target for a module, then adds a corresponding Python
extension module.

``cython_modules``
  The list of modules to build.

``linked_libraries``
  The list of libraries that need to be linked into all modules. In RAPIDS,
  this list usually contains (at minimum) the corresponding C++ libraries.

#]=======================================================================]
function(add_cython_modules cython_modules linked_libraries)
  foreach(cython_module ${cython_modules})
    add_cython_target(${cython_module} CXX PY3)
    add_library(${cython_module} MODULE ${cython_module})
    python_extension_module(${cython_module})

    # To avoid libraries being prefixed with "lib".
    set_target_properties(${cython_module} PROPERTIES PREFIX "")
    foreach(lib ${linked_libraries})
      target_link_libraries(${cython_module} ${lib})
    endforeach()

    # Compute the install directory relative to the source and rely on installs being relative to
    # the CMAKE_PREFIX_PATH for e.g. editable installs.
    cmake_path(RELATIVE_PATH CMAKE_CURRENT_SOURCE_DIR BASE_DIRECTORY ${raft-python_SOURCE_DIR}
               OUTPUT_VARIABLE install_dst)
    install(TARGETS ${cython_module} DESTINATION ${install_dst})
  endforeach(cython_module ${cython_sources})
endfunction()
