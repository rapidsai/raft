# RAFT Build and Development Guide

- [Building and installing RAFT](#build_install)
    - [CUDA/GPU Requirements](#cuda_gpu_req)
    - [Build Dependencies](#required_depenencies)
    - [Header-only C++](#install_header_only_cpp)
    - [C++ Shared Libraries](#shared_cpp_libs)
    - [Googletests](#gtests)
    - [C++ Using Cmake](#cpp_using_cmake)
    - [Python](#python)
- [Using RAFT in downstream projects](#use_raft)
    - [Cmake Header-only Integration](#cxx_integration)
    - [Using Shared Libraries in Cmake](#use_shared_libs)
    - [Building RAFT C++ from source](#build_cxx_source)
    - [Python/Cython Integration](#py_integration)

## <a id="build_install"></a>Building and installing RAFT

### <a id="cuda_gpu_req"></a>CUDA/GPU Requirements
- CUDA Toolkit 11.0+
- NVIDIA driver 450.80.02+
- Pascal architecture of better (Compute capability >= 6.0)

### <a id="required_dependencies"></a>Build Dependencies

Below are the dependencies for building RAFT from source. Many of these dependencies can be installed with [conda](https://anaconda.org) or [rapids-cpm](https://github.com/rapidsai/rapids-cmake#cpm).

#### Required
- [Thrust](https://github.com/NVIDIA/thrust) v1.15 / [CUB](https://github.com/NVIDIA/cub)
- [RMM](https://github.com/rapidsai/rmm)
- [Libcu++](https://github.com/NVIDIA/libcudacxx) v1.7.0
- [mdspan](https://github.com/rapidsai/mdspan)
  
#### Optional
- [cuCollections](https://github.com/NVIDIA/cuCollections) - Used in `raft::sparse::distance` API
- [FAISS](https://github.com/facebookresearch/faiss) v1.7.0 - Used in `raft::spatial::knn` API
- [NCCL](https://github.com/NVIDIA/nccl) - Used in `raft::comms` API and needed to build `Pyraft`
- [UCX](https://github.com/openucx/ucx) - Used in `raft::comms` API and needed to build `Pyraft`
- [Googletest](https://github.com/google/googletest) - Needed to build tests
- [Googlebench](https://github.com/google/benchmark) - Needed to build benchmarks
- [Doxygen](https://github.com/doxygen/doxygen) - Needed to build docs


C++ RAFT is a header-only library with the option of building shared libraries with template instantiations for common types to speed up compile times for larger projects.

The recommended way to build and install RAFT is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python code and provides options for building and installing the core headers, tests, benchmarks, and individual shared libraries.

### <a id="install_header_only_cpp"></a>Header-only C++

`build.sh` uses [rapids-cmake](https://github.com/rapidsai/rapids-cmake), which will automatically download any dependencies. It's important to note that while all the headers will be installed and available, some parts of the RAFT API depend on libraries like `FAISS`, which will need to be explicitly enabled in `build.sh`.

The following example will download the needed dependencies and install the `raft core` headers in `$INSTALL_PREFIX/include/raft`. The `raft core` headers are a subset of the RAFT headers which are safe to install and expose through public APIs in consuming projects as they require only RMM and the libraries provided by the CUDA toolkit.
```bash
./build.sh libraft
```

###<a id="shared_cpp_libs"></a>C++ Shared Libraries (optional)

Shared libraries can be built to speed up compile times for larger libraries which may heavily utilize some of the APIs. These shared libraries can also significantly improve re-compile times while developing against the APIs. Build all the shared libraries by passing `--compile-libs` flag to `build.sh`:

```bash
./build.sh libraft --compile-libs
```

Individual shared libraries have their own flags and multiple can be used (though currently only the `nn` and `distance` packages contain shared libraries):
```bash
./build.sh libraft --compile-nn --compile-dist
```

The `--install` flag can be passed to `build.sh` to install the shared libraries.

###<a id="gtests"></a>Tests

Compile the tests using the `tests` target in `build.sh`:
```bash
./build.sh libraft tests --compile-libs
```

To run C++ tests:

```bash
./cpp/build/test_raft
```

###<a id="benchmarks"></a>Benchmarks

Compile the benchmarks using the `bench` target in `build.sh`:
```bash
./build.sh libraft bench
```

To run the benchmarks:

```bash
./cpp/build/bench_raft
```

### <a id="cpp_using_cmake"></a>C++ Using Cmake

Use `CMAKE_INSTALL_PREFIX` to install RAFT into a specific location. The snippet below will install it into the current conda environment:
```bash
cd cpp
mkdir build
cd build
cmake -D BUILD_TESTS=ON -DRAFT_COMPILE_LIBRARIES=ON -DRAFT_ENABLE_NN_DEPENDENCIES=ON  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make -j<parallel_level> install
```

RAFT's cmake has the following configurable flags available:.

| Flag | Possible Values | Default Value | Behavior |
| --- | --- | --- | --- |
| BUILD_TESTS | ON, OFF | ON | Compile Googletests |
| BUILD_BENCH | ON, OFF | ON | Compile benchmarks |
| RAFT_COMPILE_LIBRARIES | ON, OFF | OFF | Compiles all `libraft` shared libraries (these are required for Googletests) |
| RAFT_COMPILE_NN_LIBRARY | ON, OFF | ON | Compiles the `libraft-nn` shared library |
| RAFT_COMPILE_DIST_LIBRARY | ON, OFF | ON | Compiles the `libraft-distance` shared library |
| RAFT_ENABLE_NN_DEPENDENCIES | ON, OFF | OFF | Searches for dependencies of nearest neighbors API, such as FAISS, and compiles them if not found. Needed for `raft::spatial::knn` |
| RAFT_ENABLE_cuco_DEPENDENCY | ON, OFF | ON | Enables the cuCollections dependency used by `raft::sparse::distance` |
| RAFT_ENABLE_nccl_DEPENDENCY | ON, OFF | OFF | Enables NCCL dependency used by `raft::comms` and needed to build `pyraft` |
| RAFT_ENABLE_ucx_DEPENDENCY | ON, OFF | OFF | Enables UCX dependency used by `raft::comms` and needed to build `pyraft` |
| RAFT_USE_FAISS_STATIC | ON, OFF | OFF | Statically link FAISS into `libraft-nn` | 
| RAFT_STATIC_LINK_LIBRARIES | ON, OFF | ON | Build static link libraries instead of shared libraries |
| DETECT_CONDA_ENV | ON, OFF | ON | Enable detection of conda environment for dependencies |
| NVTX | ON, OFF | OFF | Enable NVTX Markers |
| CUDA_ENABLE_KERNELINFO | ON, OFF | OFF | Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer` |
| CUDA_ENABLE_LINEINFO  | ON, OFF | OFF | Enable the -lineinfo option for nvcc |
| CUDA_STATIC_RUNTIME | ON, OFF | OFF | Statically link the CUDA runtime |

Currently, shared libraries are provided for the `libraft-nn` and `libraft-distance` components. The `libraft-nn` component depends upon [FAISS](https://github.com/facebookresearch/faiss) and the `RAFT_ENABLE_NN_DEPENDENCIES` option will build it from source if it is not already installed.

### <a id="python"></a>Python

Conda environment scripts are provided for installing the necessary dependencies for building and using the Python APIs. It is preferred to use `mamba`, as it provides significant speedup over `conda`. The following example will install create and install dependencies for a CUDA 11.5 conda environment:

```bash
conda env create --name raft_env -f conda/environments/raft_dev_cuda11.5.yml
conda activate raft_env
```

The Python API can be built using the `build.sh` script:

```bash
./build.sh pyraft
```

`setup.py` can also be used to build the Python API manually:
```bash
cd python
python setup.py build_ext --inplace
python setup.py install
```

To run the Python tests:
```bash
cd python
py.test -s -v raft
```

## <a id="use_raft"></a>Using RAFT in downstream projects

### <a id="cxx_integration"></a>C++ header-only integration using cmake

When the needed [build dependencies](#required_depenencies) are already satisfied, RAFT can be trivially integrated into downstream projects by cloning the repository and adding `cpp/include` from RAFT to the include path:
```cmake
set(RAFT_GIT_DIR ${CMAKE_CURRENT_BINARY_DIR}/raft CACHE STRING "Path to RAFT repo")
ExternalProject_Add(raft
  GIT_REPOSITORY    git@github.com:rapidsai/raft.git
  GIT_TAG           branch-22.04
  PREFIX            ${RAFT_GIT_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   "")
set(RAFT_INCLUDE_DIR ${RAFT_GIT_DIR}/src/raft/cpp/include CACHE STRING "RAFT include variable")
```

The RAFT headers are broken down into two different include paths so that core headers can be isolated between projects while public API headers can be installed globally, exposed to users through public APIs, and shared across projects.
- `cpp/include/raft` contains public API headers that require only rmm and the cudatoolkit libraries. These are safe to expose on public APIs and don't require `nvcc` to compile. 
- `cpp/include/raft` contains the core of the RAFT header-only library, containing primitives, algorithms, and other tools.

If RAFT has already been installed, such as by using the `build.sh` script, 
Use `find_package(raft)` and the `raft::raft` target if using RAFT to interact only with the public APIs of consuming projects.

Use `find_package(raft COMPONENTS core)` and both the `raft::raft` and `raft::core` targets when building a library that uses headers in `include/raft`.

### <a id="use_shared_libs"></a>Using pre-compiled shared libraries

Use `find_package(raft COMPONENTS core nn distance)` to enable the shared libraries and pass dependencies through separate targets for each component. In this example, the `raft::distance` and `raft::nn` targets will be available in addition to `raft::raft` and `raft::core` for configuring linking paths. These targets will also pass through any transitive dependencies (such as FAISS for the `nn` package).

The pre-compiled libraries contain template specializations for commonly used types. In order to use the symbols in the pre-compiled libraries, the compiler needs to be told not to instantiate templates that are already contained in the shared libraries. By convention, these header files are named `spectializations.hpp` and located in the base directory for the packages that contain specializations.

The following example ignores the pre-compiled templates for the `libraft-distance` API so the symbols from pre-compiled shared library will be used:
```c++
#include <raft/distance/distance.hpp>
#include <raft/distance/specializations.hpp>
```

### <a id="build_cxx_source"></a>Building RAFT C++ from source in cmake

RAFT uses the [RAPIDS cmake](https://github.com/rapidsai/rapids-cmake) library so it can be easily included into downstream projects. RAPIDS cmake provides a convenience layer around the [Cmake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake). The following example is similar to building RAFT itself from source but allows it to be done in cmake, providing the `raft::raft` link target for `include/raft` headers and `raft::core` for the `include/raft` headers. The `COMPILE_LIBRARIES` option enables the building of the shared libraries.

The following `cmake` snippet enables a flexible configuration of RAFT:

```cmake

set(RAFT_VERSION "22.04")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAFT_VERSION}")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG USE_FAISS_STATIC
          COMPILE_LIBRARIES ENABLE_NN_DEPENDENCIES CLONE_ON_PIN
          USE_NN_LIBRARY USE_DISTANCE_LIBRARY)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

  #-----------------------------------------------------
  # Clone RAFT locally if PINNED_TAG has been changed
  #-----------------------------------------------------
  if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${RAFT_VERSION}")
    message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
    set(CPM_DOWNLOAD_raft ON)
    set(CMAKE_IGNORE_PATH "${CMAKE_INSTALL_PREFIX}/include/raft;${CMAKE_IGNORE_PATH})
  endif()

  #-----------------------------------------------------
  # Add components
  #-----------------------------------------------------

  string(APPEND RAFT_COMPONENTS "core")
  if(PKG_USE_NN_LIBRARY)
    string(APPEND RAFT_COMPONENTS " nn")
  endif()

  if(PKG_USE_DISTANCE_LIBRARY)
    string(APPEND RAFT_COMPONENTS " distance")
  endif()

  #-----------------------------------------------------
  # Invoke CPM find_package()
  #-----------------------------------------------------

  rapids_cpm_find(raft ${PKG_VERSION}
          GLOBAL_TARGETS      raft::raft
          BUILD_EXPORT_SET    projname-exports
          INSTALL_EXPORT_SET  projname-exports
          CPM_ARGS
          GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
          GIT_TAG        ${PKG_PINNED_TAG}
          SOURCE_SUBDIR  cpp
          FIND_PACKAGE_ARGUMENTS "COMPONENTS ${RAFT_COMPONENTS}"
          OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_BENCH OFF"
          "RAFT_ENABLE_NN_DEPENDENCIES ${PKG_ENABLE_NN_DEPENDENCIES}"
          "RAFT_USE_FAISS_STATIC ${PKG_USE_FAISS_STATIC}"
          "RAFT_COMPILE_LIBRARIES ${PKG_COMPILE_LIBRARIES}"
  )

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${RAFT_VERSION}.00
        FORK             ${RAFT_FORK}
        PINNED_TAG       ${RAFT_PINNED_TAG}

        # When PINNED_TAG above doesn't match cuml,
        # force local raft clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ON

        COMPILE_LIBRARIES      NO
        USE_NN_LIBRARY         NO
        USE_DISTANCE_LIBRARY   NO
        ENABLE_NN_DEPENDENCIES NO  # This builds FAISS if not installed
        USE_FAISS_STATIC       NO
)
```

### <a id="py_integration"></a>Python/Cython Integration

Once installed, RAFT's Python library can be imported and used directly.
