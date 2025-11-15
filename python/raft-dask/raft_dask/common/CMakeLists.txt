# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

option(USE_NCCL_RUNTIME_WHEEL "Use the NCCL wheel at runtime instead of the system library" OFF)

set(cython_sources comms_utils.pyx nccl.pyx)
set(linked_libraries raft::raft raft::distributed)
rapids_cython_create_modules(
  SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" CXX
)

if(USE_NCCL_RUNTIME_WHEEL)
  set(rpaths "$ORIGIN/../../nvidia/nccl/lib")
  foreach(tgt IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
    set_property(
      TARGET ${tgt}
      PROPERTY INSTALL_RPATH ${rpaths}
      APPEND
    )
  endforeach()
endif()
