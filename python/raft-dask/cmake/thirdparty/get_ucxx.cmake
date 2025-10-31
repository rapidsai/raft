#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_ucxx)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    set(options UCXX_STATIC)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    set(BUILD_UCXX_SHARED ON)
    if(PKG_UCXX_STATIC)
      set(BUILD_UCXX_SHARED OFF)
    endif()

    rapids_cpm_find(ucxx ${PKG_VERSION}
            GLOBAL_TARGETS         ucxx::ucxx ucxx::python
            BUILD_EXPORT_SET       raft-distributed-exports
            INSTALL_EXPORT_SET     raft-distributed-exports
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/ucxx.git
            GIT_TAG                ${PKG_PINNED_TAG}
            SOURCE_SUBDIR          cpp
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            OPTIONS
              "BUILD_TESTS OFF"
              "BUILD_BENCH OFF"
              "UCXX_ENABLE_PYTHON ON"
              "UCXX_ENABLE_RMM ON"
              "BUILD_SHARED_LIBS ${BUILD_UCXX_SHARED}"
        )

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different ucxx locally, set the CMake variable
# CPM_ucxx_SOURCE=/path/to/local/ucxx
find_and_configure_ucxx(VERSION  0.47
        FORK             rapidsai
        PINNED_TAG       branch-0.47
        EXCLUDE_FROM_ALL YES
        UCXX_STATIC      ${RAFT_DASK_UCXX_STATIC}
    )
