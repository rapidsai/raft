# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Set the list of Cython files to build
set(cython_sources rmat_rectangular_generator.pyx)

# TODO: should finally be replaced with 'compiled' library to be more generic, when that is
# available
set(linked_libraries raft::raft raft::compiled)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX random_
)
