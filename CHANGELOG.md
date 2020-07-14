# RAFT 0.15.0 (Date TBD)

## New Features
- PR #12: Spectral clustering.
- PR #7: Migrating cuml comms -> raft comms_t
- PR #15: add exception based error handling macros
- PR #29: Add ceildiv functionality

## Improvements
- PR #13: Add RMM_INCLUDE and RMM_LIBRARY options to allow linking to non-conda RMM
- PR #22: Preserve order in comms workers for rank initialization

## Bug Fixes
- PR #17: Make destructor inline to avoid redeclaration error
- PR #25: Fix bug in handle_t::get_internal_streams
- PR #26: Fix bug in RAFT_EXPECTS (add parentheses surrounding cond)
- PR #34 Fix issue with incorrect docker image being used in local build script

# RAFT 0.14.0 (Date TBD)

## New Features
- Initial RAFT version
- PR #3: defining raft::handle_t, device_buffer, host_buffer, allocator classes

## Bug Fixes
- PR #5: Small build.sh fixes
