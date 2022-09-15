# Developer Guide

## Local Development

Devloping features and fixing bugs for the RAFT library itself is straightforward and only requires building and installing the relevant RAFT artifacts. 

The process for working on a CUDA/C++ feature which might span RAFT and one or more consuming libraries can vary slightly depending on whether the consuming project relies on a source build (as outlined in the [BUILD](BUILD.md#install_header_only_cpp) docs). In such a case, the option `CPM_raft_SOURCE=/path/to/raft/source` can be passed to the cmake of the consuming project in order to build the local RAFT from source. The PR with relevant changes to the consuming project can also pin the RAFT version temporarily by explicitly changing the `FORK` and `PINNED_TAG` arguments to the RAFT branch containing their changes when invoking `find_and_configure_raft`.  The pin should be reverted after the changed is merged to the RAFT project and before it is merged to the dependent project(s) downstream.

If building a feature which spans projects and not using the source build in cmake, the RAFT changes (both C++ and Python) will need to be installed into the environment of the consuming project before they can be used. The ideal integration of RAFT into consuming projects will enable both the source build in the consuming project only for this case but also rely on a more stable packaging (such as conda packaging) otherwise. 

## API stability

Since RAFT is a core library with multiple consumers, it's important that the public APIs maintain stability across versions and any changes to them are done with caution, adding new functions and deprecating the old functions over a couple releases as necessary.

The public APIs should be lightweight wrappers around calls to private APIs inside the `detail` namespace. 

## Common Design Considerations

1. Use the `hpp` extension for files which can be compiled with `gcc` against the CUDA-runtime. Use the `cuh` extension for files which require `nvcc` to be compiled. `hpp` can also be used for functions marked `__host__ __device__` only if proper checks are in place to remove the `__device__` designation when not compiling with `nvcc`. 

2. When additional classes, structs, or general POCO types are needed to be used for representing data in the public API, place them in a new file called `<primitive_name>_types.hpp`. This tells users they are safe to expose these types on their own public APIs without bringing in device code. At a minimum, the definitions for these types, at least, should not require `nvcc`. In general, these classes should only store very simple state and should not perform their own computations. Instead, new functions should be exposed on the public API which accept these objects, reading or updating their state as necessary. 

3. Documentation for public APIs should be well documented, easy to use, and it is highly preferred that they include usage instructions.

4. Before creating a new primitive, check to see if one exists already. If one exists but the API isn't flexible enough to include your use-case, consider first refactoring the existing primitive. If that is not possible without an extreme number of changes, consider how the public API could be made more flexible. If the new primitive is different enough from all existing primitives, consider whether an existing public API could invoke the new primitive as an option or argument. If the new primitive is different enough from what exists already, add a header for the new public API function to the appropriate subdirectory and namespace.

## Testing

It's important for RAFT to maintain a high test coverage in order to minimize the potential for downstream projects to encounter unexpected build or runtime behavior as a result of changes. A well-defined public API can help maintain compile-time stability but means more focus should be placed on testing the functional requirements and verifying execution on the various edge cases within RAFT itself. Ideally, bug fixes and new features should be able to be made to RAFT independently of the consuming projects.


## Documentation

Public APIs always require documentation, since those will be exposed directly to users. In addition to summarizing the purpose of each class / function in the public API, the arguments (and relevant templates) should be documented along with brief usage examples.