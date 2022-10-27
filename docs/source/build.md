# Install Guide

## Building and installing RAFT

### CUDA/GPU Requirements
- CUDA Toolkit 11.0+
- NVIDIA driver 450.80.02+
- Pascal architecture of better (compute capability >= 6.0)

### Build Dependencies

In addition to the libraries included with cudatoolkit 11.0+, there are some other dependencies below for building RAFT from source. Many of the dependencies are optional and depend only on the primitives being used. All of these can be installed with cmake or [rapids-cpm](https://github.com/rapidsai/rapids-cmake#cpm) and many of them can be installed with [conda](https://anaconda.org).

#### Required
- [RMM](https://github.com/rapidsai/rmm) corresponding to RAFT version.
- [Thrust](https://github.com/NVIDIA/thrust) v1.17 / [CUB](https://github.com/NVIDIA/cub)

#### Optional
- [cuCollections](https://github.com/NVIDIA/cuCollections) - Used in `raft::sparse::distance` API.
- [Libcu++](https://github.com/NVIDIA/libcudacxx) v1.7.0
- [FAISS](https://github.com/facebookresearch/faiss) v1.7.0 - Used in `raft::spatial::knn` API and needed to build tests.
- [NCCL](https://github.com/NVIDIA/nccl) - Used in `raft::comms` API and needed to build `raft-dask`
- [UCX](https://github.com/openucx/ucx) - Used in `raft::comms` API and needed to build `raft-dask`
- [Googletest](https://github.com/google/googletest) - Needed to build tests
- [Googlebench](https://github.com/google/benchmark) - Needed to build benchmarks
- [Doxygen](https://github.com/doxygen/doxygen) - Needed to build docs

C++ RAFT is a header-only library but provides the option of building shared libraries with template instantiations for common types to speed up compile times for larger projects.

The recommended way to build and install RAFT is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python artifacts and provides options for building and installing the headers, tests, benchmarks, and individual shared libraries.

### Header-only C++

`build.sh` uses [rapids-cmake](https://github.com/rapidsai/rapids-cmake), which will automatically download any dependencies which are not already installed. It's important to note that while all the headers will be installed and available, some parts of the RAFT API depend on libraries like `FAISS`, which will need to be explicitly enabled in `build.sh`.

The following example will download the needed dependencies and install the RAFT headers into `$INSTALL_PREFIX/include/raft`. The `--install` flag can be omitted to just have the build download the needed dependencies. Since RAFT is primarily used at build-time, the dependencies will never be installed by the RAFT build, with the exception of building FAISS statically into the shared libraries.
```bash
./build.sh libraft --install
```

### C++ Shared Libraries (optional)

For larger projects which make heavy use of the pairwise distances or nearest neighbors APIs, shared libraries can be built to speed up compile times. These shared libraries can also significantly improve re-compile times both while developing RAFT and developing against the APIs. Build all of the available shared libraries by passing `--compile-libs` flag to `build.sh`:
```bash
./build.sh libraft --compile-libs
```

Individual shared libraries have their own flags and multiple can be used (though currently only the `nn` and `distance` packages contain shared libraries):
```bash
./build.sh libraft --compile-nn --compile-dist
```

Add the `--install` flag to the above example to also install the shared libraries into `$INSTALL_PREFIX/lib`.

### ccache and sccache

`ccache` and `sccache` can be used to better cache parts of the build when rebuilding frequently, such as when working on a new feature. You can also use `ccache` or `sccache` with `build.sh`:

```bash
./build.sh libraft --cache-tool=ccache
```

### Tests

Compile the tests using the `tests` target in `build.sh`.

```bash
./build.sh libraft tests
```

Test compile times can be improved significantly by using the optional shared libraries. If installed, they will be used automatically when building the tests but `--compile-libs` can be used to add additional compilation units and compile them with the tests.

```bash
./build.sh libraft tests --compile-libs
```

The tests are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_TEST`.

For example, to run the distance tests:
```bash
./cpp/build/DISTANCE_TEST
```

It can take sometime to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the `--limit-tests` option in `build.sh`:

```bash
./build.sh libraft tests --limit-tests=NEIGHBORS_TEST;DISTANCE_TEST;MATRIX_TEST
```

### Benchmarks

The benchmarks are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_BENCH`.
```bash
./build.sh libraft bench
```

It can take sometime to compile all of the benchmarks. You can build individual benchmarks by providing a semicolon-separated list to the `--limit-bench` option in `build.sh`:

```bash
./build.sh libraft bench --limit-bench=NEIGHBORS_BENCH;DISTANCE_BENCH;LINALG_BENCH
```

### C++ Using Cmake

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
| raft_FIND_COMPONENTS | nn distance | | Configures the optional components as a space-separated list |
| RAFT_COMPILE_LIBRARIES | ON, OFF | OFF | Compiles all `libraft` shared libraries (these are required for Googletests) |
| RAFT_COMPILE_NN_LIBRARY | ON, OFF | OFF | Compiles the `libraft-nn` shared library |
| RAFT_COMPILE_DIST_LIBRARY | ON, OFF | OFF | Compiles the `libraft-distance` shared library |
| RAFT_ENABLE_NN_DEPENDENCIES | ON, OFF | OFF | Searches for dependencies of nearest neighbors API, such as FAISS, and compiles them if not found. Needed for `raft::spatial::knn` |
| RAFT_USE_FAISS_STATIC | ON, OFF | OFF | Statically link FAISS into `libraft-nn` |
| RAFT_STATIC_LINK_LIBRARIES | ON, OFF | ON | Build static link libraries instead of shared libraries |
| DETECT_CONDA_ENV | ON, OFF | ON | Enable detection of conda environment for dependencies |
| NVTX | ON, OFF | OFF | Enable NVTX Markers |
| CUDA_ENABLE_KERNELINFO | ON, OFF | OFF | Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer` |
| CUDA_ENABLE_LINEINFO  | ON, OFF | OFF | Enable the -lineinfo option for nvcc |
| CUDA_STATIC_RUNTIME | ON, OFF | OFF | Statically link the CUDA runtime |

Currently, shared libraries are provided for the `libraft-nn` and `libraft-distance` components. The `libraft-nn` component depends upon [FAISS](https://github.com/facebookresearch/faiss) and the `RAFT_ENABLE_NN_DEPENDENCIES` option will build it from source if it is not already installed.

### Python

Conda environment scripts are provided for installing the necessary dependencies for building and using the Python APIs. It is preferred to use `mamba`, as it provides significant speedup over `conda`. In addition you will have to manually install `nvcc` as it will not be installed as part of the conda environment. The following example will install create and install dependencies for a CUDA 11.5 conda environment:

```bash
mamba env create --name raft_env_name -f conda/environments/raft_dev_cuda11.5.yml
mamba activate raft_env_name
```

The Python APIs can be built and installed using the `build.sh` script:

```bash
# to build pylibraft
./build.sh libraft pylibraft --install --compile-libs
# to build raft-dask
./build.sh libraft raft-dask --install --compile-libs
```

`setup.py` can also be used to build the Python APIs manually:

```
cd python/raft-dask
python setup.py build_ext --inplace
python setup.py install

cd python/pylibraft
python setup.py build_ext --inplace
python setup.py install
```

To run the Python tests:
```bash
cd python/raft-dask
py.test -s -v

cd python/pylibraft
py.test -s -v
```

### Documentation

The documentation requires that the C++ headers and python packages have been built and installed.

The following will build the docs along with the C++ and Python packages:

```
./build.sh libraft pylibraft raft-dask docs --compile-libs --install
```



## Using RAFT in downstream projects

There are two different strategies for including RAFT in downstream projects, depending on whether or not the required dependencies are already installed and available on the `lib` and `include` paths.

### C++ header-only integration using cmake

When the needed [build dependencies](#required_depenencies) are already satisfied, RAFT can be trivially integrated into downstream projects by cloning the repository and adding `cpp/include` from RAFT to the include path:
```cmake
set(RAFT_GIT_DIR ${CMAKE_CURRENT_BINARY_DIR}/raft CACHE STRING "Path to RAFT repo")
ExternalProject_Add(raft
  GIT_REPOSITORY    git@github.com:rapidsai/raft.git
  GIT_TAG           branch-22.10
  PREFIX            ${RAFT_GIT_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   "")
set(RAFT_INCLUDE_DIR ${RAFT_GIT_DIR}/raft/cpp/include CACHE STRING "RAFT include variable")
```

If RAFT has already been installed, such as by using the `build.sh` script, use `find_package(raft)` and the `raft::raft` target if using RAFT to interact only with the public APIs of consuming projects.

### Using pre-compiled shared libraries

Use `find_package(raft COMPONENTS nn distance)` to enable the shared libraries and transitively pass dependencies through separate targets for each component. In this example, the `raft::distance` and `raft::nn` targets will be available for configuring linking paths in addition to `raft::raft`. These targets will also pass through any transitive dependencies (such as FAISS for the `nn` package).

The pre-compiled libraries contain template specializations for commonly used types, such as single- and double-precision floating-point. In order to use the symbols in the pre-compiled libraries, the compiler needs to be told not to instantiate templates that are already contained in the shared libraries. By convention, these header files are named `specializations.hpp` and located in the base directory for the packages that contain specializations.

The following example tells the compiler to ignore the pre-compiled templates for the `libraft-distance` API so any symbols already compiled into pre-compiled shared library will be used instead:
```c++
#include <raft/distance/distance.cuh>
#include <raft/distance/specializations.cuh>
```

### Building RAFT C++ from source in cmake

RAFT uses the [RAPIDS-CMake](https://github.com/rapidsai/rapids-cmake) library so it can be more easily included into downstream projects. RAPIDS cmake provides a convenience layer around the [CMake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake).

The following example is similar to invoking `find_package(raft)` but uses `rapids_cpm_find`, which provides a richer and more flexible configuration landscape by using CPM to fetch any dependencies not already available to the build. The `raft::raft` link target will be made available and it's recommended that it be used as a `PRIVATE` link dependency in downstream projects. The `COMPILE_LIBRARIES` option enables the building the shared libraries.

The following `cmake` snippet enables a flexible configuration of RAFT:

```cmake

set(RAFT_VERSION "22.10")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAFT_VERSION}")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG USE_FAISS_STATIC
          COMPILE_LIBRARIES ENABLE_NN_DEPENDENCIES CLONE_ON_PIN
          USE_NN_LIBRARY USE_DISTANCE_LIBRARY
          ENABLE_thrust_DEPENDENCY)
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
          "RAFT_ENABLE_thrust_DEPENDENCY ${PKG_ENABLE_thrust_DEPENDENCY}"
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

        COMPILE_LIBRARIES        NO
        USE_NN_LIBRARY           NO
        USE_DISTANCE_LIBRARY     NO
        ENABLE_NN_DEPENDENCIES   NO  # This builds FAISS if not installed
        USE_FAISS_STATIC         NO
        ENABLE_thrust_DEPENDENCY YES
)
```

If using the nearest neighbors APIs without the shared libraries, set `ENABLE_NN_DEPENDENCIES=ON` and keep `USE_NN_LIBRARY=OFF`

### Python/Cython Integration

Once installed, RAFT's Python library can be added to downstream conda recipes, imported and used directly.
