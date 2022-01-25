# RAFT Build and Development Guide

- [Building and running tests](#building-and-running-tests)
- [Usage of RAFT by downstream projects](#usage-of-raft-by-downstream-projects)
    - [C++ Integration](#c-integration)
    - [Python/Cython Integration](#pythoncython-integration)
- [CI Process](#ci-process)
- [Developer Guide](#developer-guide)
    - [Local Development](#local-development)
    - [Submitting PRs](#submitting-prs)

## Building and installing RAFT

C++ RAFT is a header-only library but provides the option of building shared libraries with template instantiations for common types to speed up compile times for larger projects. The recommended way to build and install RAFT is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python code and provides options for building and installing the shared libraries.

To run C++ tests:

```bash
./test_raft
```

To run Python tests, if `install` setup.py target is not run:

```bash
cd python
python -m pytest raft
```

To build manually, you can also use `CMake` and setup.py directly.

For C++, the `RAFT_COMPILE_LIBRARIES` option can be used to compile the shared libraries. Shared libraries are provided for the `nn` and `distance` packages currently. The `nn` package requires FAISS, which will be built from source if it is not already installed. FAISS can optionally be statically compiled into the `nn` shared library with the `RAFT_USE_FAISS_STATIC` option.

To install RAFT into a specific location, use `CMAKE_INSTALL_PREFIX`. The snippet below will install it into the current conda environment.
```bash
cd cpp
mkdir build
cd build
cmake -DRAFT_COMPILE_LIBRARIES=ON -DRAFT_USE_FAISS_STATIC=OFF  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make install
```

For python:

```bash
cd python
python setup.py build_ext --inplace
python setup.py install
```

## Using RAFT in downstream projects

### C++ Integration

Use RAFT in cmake projects with `find_package(raft)` for header-only operation and the `raft::raft` target will be available for configuring linking and `RAFT_INCLUDE_DIR` will be available for includes. Note that if any packages are used which require downstream dependencies, such as the `nn` package requiring FAISS, these dependencies will have be installed and configured in cmake independently.

Use `find_package(raft COMPONENTS nn, distance)` to enable the shared libraries and pass dependencies through separate targets for each component. In this example, `raft::distance` and `raft::nn` targets will be available for configuring linking paths. These targets will also pass through any transitive dependencies (such as FAISS in the case of the `nn` package).

### Building RAFT C++ from source

RAFT uses the [RAPIDS cmake](https://github.com/rapidsai/rapids-cmake) library, so it can be easily included into downstream projects. RAPIDS cmake provides a convenience layer around the [Cmake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake). The following example is similar to building RAFT itself from source but allows it to be done in cmake, providing the `raft::raft` target for includes by default. The `COMPILE_LIBRARIES` option enables the building of the shared libraries 

```cmake
function(find_and_configure_raft)

  set(oneValueArgs VERSION FORK PINNED_TAG USE_RAFT_NN USE_FAISS_STATIC COMPILE_LIBRARIES)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

  rapids_cpm_find(raft ${PKG_VERSION}
          GLOBAL_TARGETS      raft::raft
          BUILD_EXPORT_SET    proj-exports
          INSTALL_EXPORT_SET  proj-exports
          CPM_ARGS
          GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
          GIT_TAG        ${PKG_PINNED_TAG}
          SOURCE_SUBDIR  cpp
          FIND_PACKAGE_ARGUMENTS "COMPONENTS ${RAFT_COMPONENTS}"
          OPTIONS
          "BUILD_TESTS OFF"
          "RAFT_USE_FAISS_STATIC ${PKG_USE_FAISS_STATIC}"
          "NVTX ${NVTX}"
          "RAFT_COMPILE_LIBRARIES ${COMPILE_LIBRARIES}"
          
  )

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    22.02.00
        FORK             rapidsai
        PINNED_TAG       branch-22.02
        USE_RAFT_NN       NO
        USE_FAISS_STATIC  NO
        COMPILE_LIBRARIES NO
)
```

### Python/Cython Integration

Once installed, RAFT's Python library can be imported and used directly.