#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

function(find_and_configure_diskann)
    include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/diskann_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(diskann version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(diskann ${version} patch_command)

  rapids_cpm_find(diskann ${version}
          GLOBAL_TARGETS diskann::diskann
          CPM_ARGS
          GIT_REPOSITORY ${repository}
          GIT_TAG ${tag}
          GIT_SHALLOW ${patch_command}
          OPTIONS
          "PYBIND OFF"
          "UNIT_TEST OFF"
          "RESTAPI OFF"
          "PORTABLE OFF"
          "-DOMP_PATH /raid/tarangj/miniconda3/envs/all_cuda-122_arch-x86_64/lib/libiomp5.so"
          "-DMKL_PATH /raid/tarangj/miniconda3/envs/all_cuda-122_arch-x86_64/lib"
          )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(diskann)
    
  if(NOT TARGET diskann::diskann)
      target_include_directories(diskann INTERFACE "$<BUILD_INTERFACE:${diskann_SOURCE_DIR}/include>")
      add_library(diskann::diskann ALIAS diskann)
  endif()
endfunction()
find_and_configure_diskann()