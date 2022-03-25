# Developer Guide

## Local Development

Devloping features and fixing bugs for the RAFT library itself is straightforward and only requires building and installing the relevant RAFT artifacts. 

The process for working on a CUDA/C++ feature which spans RAFT and one or more consumers can vary slightly depending on whether the consuming project relies on a source build (as outlined in the [BUILD](BUILD.md#install_header_only_cpp) docs). In such a case, the option `CPM_raft_SOURCE=/path/to/raft/source` can be passed to the cmake of the consuming project in order to build the local RAFT from source. The PR with relevant changes to the consuming project can also pin the RAFT version temporarily by explicitly changing the `FORK` and `PINNED_TAG` arguments to the RAFT branch containing their changes when invoking `find_and_configure_raft`.  The pin should be reverted after the changed is merged to the RAFT project and before it is merged to the dependent project(s) downstream.

If building a feature which spans projects and not using the source build in cmake, the RAFT changes (both C++ and Python) will need to be installed into the environment of the consuming project before they can be used. The ideal integration of RAFT into consuming projects will enable both the source build in the consuming project only for this case but also rely on a more stable packaging (such as conda packaging) otherwise. 

## API stability

Since RAFT is a core library with multiple consumers, it's important that the public APIs maintain stability across versions and any changes to them are done with caution, adding new functions and deprecating the old functions over a couple releases as necessary.

The public APIs should be lightweight wrappers around calls to private APIs inside the `detail` namespace. 

## Testing

It's important for RAFT to maintain a high test coverage in order to minimize the potential for downstream projects to encounter unexpected build or runtime behavior as a result of changes. A well-defined public API can help maintain compile-time stability but means more focus should be placed on testing the functional requirements and verifying execution on the various edge cases within RAFT itself. Ideally, bug fixes and new features should be able to be made to RAFT independently of the consuming projects.


## Documentation

Public APIs always require documentation, since those will be exposed directly to users. In addition to summarizing the purpose of each class / function in the public API, the arguments (and relevant templates) should be documented along with brief usage examples.