# RAFT 0.16.0 (Date TBD)

## New Features

- PR #63: Adding MPI comms implementation

## Improvements
- PR #59: Adding csrgemm2 to cusparse_wrappers.h
- PR #61: Add cusparsecsr2dense to cusparse_wrappers.h
- PR #62: Adding `get_device_allocator` to `handle.pxd`

## Bug Fixes
- PR #56: Fix compiler warnings.
- PR #60: Remove `cublas_try` from `cusolver_wrappers.h`

# RAFT 0.15.0 (Date TBD)

## New Features
- PR #12: Spectral clustering.
- PR #7: Migrating cuml comms -> raft comms_t
- PR #18: Adding commsplit to cuml communicator
- PR #15: add exception based error handling macros
- PR #29: Add ceildiv functionality
- PR #44: Add get_subcomm and set_subcomm to handle_t

## Improvements
- PR #13: Add RMM_INCLUDE and RMM_LIBRARY options to allow linking to non-conda RMM
- PR #22: Preserve order in comms workers for rank initialization
- PR #38: Remove #include <cudar_utils.h> from `raft/mr/`
- PR #39: Adding a virtual destructor to `raft::handle_t` and `raft::comms::comms_t`
- PR #37: Clean-up CUDA related utilities
- PR #41: Upgrade to `cusparseSpMV()`, alg selection, and rectangular matrices.
- PR #45: Add Ampere target to cuda11 cmake
- PR #47: Use gtest conda package in CMake/build.sh by default

## Bug Fixes
- PR #17: Make destructor inline to avoid redeclaration error
- PR #25: Fix bug in handle_t::get_internal_streams
- PR #26: Fix bug in RAFT_EXPECTS (add parentheses surrounding cond)
- PR #34: Fix issue with incorrect docker image being used in local build script
- PR #35: Remove #include <nccl.h> from `raft/error.hpp`
- PR #40: Preemptively fixed future CUDA 11 related errors.
- PR #43: Fixed CUDA version selection mechanism for SpMV.
- PR #46: Fix for cpp file extension issue (nvcc-enforced).
- PR #48: Fix gtest target names in cmake build gtest option.
- PR #49: Skip raft comms test if raft module doesn't exist

# RAFT 0.14.0 (Date TBD)

## New Features
- Initial RAFT version
- PR #3: defining raft::handle_t, device_buffer, host_buffer, allocator classes

## Bug Fixes
- PR #5: Small build.sh fixes
