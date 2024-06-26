#=============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG BUILD_STATIC_LIBS EXCLUDE_FROM_ALL ENABLE_GPU)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  rapids_find_generate_module(faiss
    HEADER_NAMES  faiss/IndexFlat.h
    LIBRARY_NAMES faiss
    )

  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/faiss_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(faiss version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(faiss ${version} patch_command)

  set(BUILD_SHARED_LIBS ON)
  if (PKG_BUILD_STATIC_LIBS)
    set(BUILD_SHARED_LIBS OFF)
    set(CPM_DOWNLOAD_faiss ON)
  endif()

  include(cmake/modules/FindAVX)
  # Link against AVX CPU lib if it exists
  set(RAFT_FAISS_OPT_LEVEL "generic")
  if(CXX_AVX2_FOUND)
    set(RAFT_FAISS_OPT_LEVEL "avx2")
  endif()

  rapids_cpm_find(faiss ${version}
    GLOBAL_TARGETS faiss faiss_avx2 faiss_gpu faiss::faiss faiss::faiss_avx2
    CPM_ARGS
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow} ${patch_command}
    EXCLUDE_FROM_ALL ${exclude}
    OPTIONS
    "FAISS_ENABLE_GPU ${PKG_ENABLE_GPU}"
    "FAISS_ENABLE_RAFT ${PKG_ENABLE_GPU}"
    "FAISS_ENABLE_PYTHON OFF"
    "FAISS_OPT_LEVEL ${RAFT_FAISS_OPT_LEVEL}"
    "FAISS_USE_CUDA_TOOLKIT_STATIC ${CUDA_STATIC_RUNTIME}"
    "BUILD_TESTING OFF"
    "CMAKE_MESSAGE_LOG_LEVEL VERBOSE"
    )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(hnswlib)

  if(TARGET faiss AND NOT TARGET faiss::faiss)
    add_library(faiss::faiss ALIAS faiss)
    # We need to ensure that faiss has all the conda information. So we use this approach so that
    # faiss will have the conda includes/link dirs
    target_link_libraries(faiss PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
  endif()
  if(TARGET faiss_avx2 AND NOT TARGET faiss::faiss_avx2)
    add_library(faiss::faiss_avx2 ALIAS faiss_avx2)
    # We need to ensure that faiss has all the conda information. So we use this approach so that
    # faiss will have the conda includes/link dirs
    target_link_libraries(faiss_avx2 PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
  endif()
  if(TARGET faiss_gpu AND NOT TARGET faiss::faiss_gpu)
    add_library(faiss::faiss_gpu ALIAS faiss_gpu)
    # We need to ensure that faiss has all the conda information. So we use this approach so that
    # faiss will have the conda includes/link dirs
    target_link_libraries(faiss_gpu PRIVATE $<TARGET_NAME_IF_EXISTS:conda_env>)
  endif()

  if(faiss_ADDED)
    rapids_export(BUILD faiss
                  EXPORT_SET faiss-targets
                  GLOBAL_TARGETS ${RAFT_FAISS_EXPORT_GLOBAL_TARGETS}
                  NAMESPACE faiss::)
  endif()

  # Need to tell CMake to rescan the link group of faiss::faiss_gpu and faiss
  # so that we get proper link order when they are static
  #
  # We don't look at the existence of `faiss_avx2` as it will always exist
  # even when CXX_AVX2_FOUND is false. In addition for arm builds the
  # faiss_avx2 is marked as `EXCLUDE_FROM_ALL` so we don't want to add
  # a dependency to it. Adding a dependency will cause it to compile,
  # and fail due to invalid compiler flags.
  if(PKG_ENABLE_GPU AND PKG_BUILD_STATIC_LIBS AND CXX_AVX2_FOUND)
    set(RAFT_FAISS_TARGETS "$<LINK_GROUP:RESCAN,$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>,faiss::faiss_avx2>" PARENT_SCOPE)
  elseif(PKG_ENABLE_GPU AND  PKG_BUILD_STATIC_LIBS)
    set(RAFT_FAISS_TARGETS "$<LINK_GROUP:RESCAN,$<LINK_LIBRARY:WHOLE_ARCHIVE,faiss_gpu>,faiss::faiss>" PARENT_SCOPE)
  elseif(CXX_AVX2_FOUND)
    set(RAFT_FAISS_TARGETS faiss::faiss_avx2 PARENT_SCOPE)
  else()
    set(RAFT_FAISS_TARGETS faiss::faiss PARENT_SCOPE)
  endif()

endfunction()


find_and_configure_faiss(
  BUILD_STATIC_LIBS ${RAFT_USE_FAISS_STATIC}
  ENABLE_GPU ${RAFT_FAISS_ENABLE_GPU}
)