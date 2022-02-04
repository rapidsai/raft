# RAFT Build and Development Guide

- [Building and installing RAFT](#build_install)
- [Using RAFT in downstream projects](#use_raft)
    - [C++ Integration](#cxx_integration)
    - [Building RAFT C++ from source](#build_cxx_source)
    - [Python/Cython Integration](#py_integration)

## <a id="build_install"></a>Building and installing RAFT

### CUDA/GPU Requirements
- CUDA 11.0+
- NVIDIA driver 450.80.02+
- Pascal architecture of better (Compute capability >= 6.0)

C++ RAFT is a header-only library but provides the option of building shared libraries with template instantiations for common types to speed up compile times for larger projects. The recommended way to build and install RAFT is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python code and provides options for building and installing the individual shared libraries.

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

For C++, the `RAFT_COMPILE_LIBRARIES` option can be used to compile the shared libraries. Shared libraries are provided for the `libraft-nn` and `libraft-distance` components currently. The `libraft-nn` component depends upon [FAISS](https://github.com/facebookresearch/faiss) and the `RAFT_ENABLE_NN_DEPENDENCIES` option will build it from source if it is not already installed. FAISS can optionally be statically compiled into the `libraft-nn` shared library with the `RAFT_USE_FAISS_STATIC` option.

To install RAFT into a specific location, use `CMAKE_INSTALL_PREFIX`. The snippet below will install it into the current conda environment.
```bash
cd cpp
mkdir build
cd build
cmake -DRAFT_COMPILE_LIBRARIES=ON -DRAFT_USE_FAISS_STATIC=OFF -DRAFT_ENABLE_NN_DEPENDENCIES=ON  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make install
```

For python:

```bash
cd python
python setup.py build_ext --inplace
python setup.py install
```

## <a id="use_raft"></a>Using RAFT in downstream projects

### <a id="cxx_integration"></a>C++ header-only integration using cmake

Use RAFT in cmake projects with `find_package(raft)` for header-only operation and the `raft::raft` target will be available for configuring linking and `RAFT_INCLUDE_DIR` will be available for includes. Note that if any packages are used which require downstream dependencies, such as the `libraft-nn` package requiring FAISS, these dependencies will have be installed and configured in cmake independently.

### <a id="build_cxx_source"></a>Using pre-compiled shared libraries

Use `find_package(raft COMPONENTS nn, distance)` to enable the shared libraries and pass dependencies through separate targets for each component. In this example, `raft::distance` and `raft::nn` targets will be available for configuring linking paths. These targets will also pass through any transitive dependencies (such as FAISS in the case of the `nn` package).

The pre-compiled libraries contain template specializations for commonly used types and require the additional include of header files with `extern template` definitions that tell the compiler not to instantiate templates that are already contained in the shared libraries. By convention, these header files are named `spectializations.hpp` and located in the base directory for the packages that contain specializations.

The following example shows how to use the `libraft-distance` API with the pre-compiled specializations:
```c++
#include <raft/distance/distance.hpp>
#include <raft/distance/specializations.hpp>
```

### <a id="build_cxx_source"></a>Building RAFT C++ from source

RAFT uses the [RAPIDS cmake](https://github.com/rapidsai/rapids-cmake) library, so it can be easily included into downstream projects. RAPIDS cmake provides a convenience layer around the [Cmake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake). The following example is similar to building RAFT itself from source but allows it to be done in cmake, providing the `raft::raft` link target and `RAFT_INCLUDE_DIR` for includes. The `COMPILE_LIBRARIES` option enables the building of the shared libraries 

```cmake
function(find_and_configure_raft)

  set(oneValueArgs VERSION FORK PINNED_TAG USE_FAISS_STATIC COMPILE_LIBRARIES ENABLE_NN_DEPENDENCIES)
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
          "RAFT_ENABLE_NN_DEPENDENCIES ${PKG_ENABLE_NN_DEPENDENCIES}"
          "RAFT_USE_FAISS_STATIC ${PKG_USE_FAISS_STATIC}"
          "RAFT_COMPILE_LIBRARIES ${PKG_COMPILE_LIBRARIES}"
  )

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    22.02.00
        FORK             rapidsai
        PINNED_TAG       branch-22.02
        COMPILE_LIBRARIES      NO
        ENABLE_NN_DEPENDENCIES NO
        USE_FAISS_STATIC       NO
)
```

### <a id="py_integration"></a>Python/Cython Integration

Once installed, RAFT's Python library can be imported and used directly.