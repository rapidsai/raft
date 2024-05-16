#=============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

function(find_and_configure_hnswlib)
  set(oneValueArgs)

  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/hnswlib_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(hnswlib version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(hnswlib ${version} patch_command)

  rapids_cpm_find(
    hnswlib ${version}
    GLOBAL_TARGETS hnswlib::hnswlib
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports
    CPM_ARGS
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow} ${patch_command}
    EXCLUDE_FROM_ALL ${exclude}
    DOWNLOAD_ONLY ON
  )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(hnswlib)

  if(NOT TARGET hnswlib::hnswlib)
    add_library(hnswlib INTERFACE )
    add_library(hnswlib::hnswlib ALIAS hnswlib)
    target_include_directories(hnswlib INTERFACE
     "$<BUILD_INTERFACE:${hnswlib_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:include>")

    if(NOT PKG_EXCLUDE_FROM_ALL)
      install(TARGETS hnswlib EXPORT hnswlib-exports)
      install(DIRECTORY "${hnswlib_SOURCE_DIR}/hnswlib/" DESTINATION include/hnswlib)

      # write install export rules
      rapids_export(
        INSTALL hnswlib
        VERSION ${PKG_VERSION}
        EXPORT_SET hnswlib-exports
        GLOBAL_TARGETS hnswlib
        NAMESPACE hnswlib::)
    endif()

    # write build export rules
    rapids_export(
      BUILD hnswlib
      VERSION ${PKG_VERSION}
      EXPORT_SET hnswlib-exports
      GLOBAL_TARGETS hnswlib
      NAMESPACE hnswlib::)

    include("${rapids-cmake-dir}/export/find_package_root.cmake")

    # When using RAFT from the build dir, ensure hnswlib is also found in RAFT's build dir. This
    # line adds `set(hnswlib_ROOT "${CMAKE_CURRENT_LIST_DIR}")` to build/raft-dependencies.cmake
    rapids_export_find_package_root(
      BUILD hnswlib [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET raft-exports
    )
  endif()
endfunction()

find_and_configure_hnswlib()
