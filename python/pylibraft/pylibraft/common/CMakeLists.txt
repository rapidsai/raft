# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Set the list of Cython files to build
set(cython_sources cuda.pyx handle.pyx mdspan.pyx interruptible.pyx)
set(linked_libraries raft::raft)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX common_
)
