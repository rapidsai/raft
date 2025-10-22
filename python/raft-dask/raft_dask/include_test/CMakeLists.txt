# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

set(cython_sources raft_include_test.pyx)
set(linked_libraries raft::raft)
rapids_cython_create_modules(
  SOURCE_FILES "${cython_sources}" LINKED_LIBRARIES "${linked_libraries}" CXX
)
