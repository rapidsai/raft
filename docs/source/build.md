# Installation

### Conda

The easiest way to install RAFT is through conda and several packages are provided.
- `libraft-headers` RAFT headers
- `libraft` (optional) shared library containing pre-compiled template specializations and runtime API.
- `pylibraft` (optional) Python wrappers around RAFT algorithms and primitives.
- `raft-dask` (optional) enables deployment of multi-node multi-GPU algorithms that use RAFT `raft::comms` in Dask clusters.

Use the following command to install all of the RAFT packages with conda (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command.
```bash
mamba install -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft
```

You can also install the conda packages individually using the `mamba` command above.

After installing RAFT, `find_package(raft COMPONENTS nn distance)` can be used in your CUDA/C++ cmake build to compile and/or link against needed dependencies in your raft target. `COMPONENTS` are optional and will depend on the packages installed.

### Pip

pylibraft and raft-dask both have experimental packages that can be [installed through pip](https://rapids.ai/pip.html#install):
```bash
pip install pylibraft-cu11 --extra-index-url=https://pypi.nvidia.com
pip install raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
```

## Building and installing RAFT

### CUDA/GPU Requirements
- cmake 3.23.1+
- GCC 9.3+ (9.5.0+ recommended)
- CUDA Toolkit 11.2+
- NVIDIA driver 450.80.02+
- Pascal architecture or better (compute capability >= 6.0)

### Build Dependencies

In addition to the libraries included with cudatoolkit 11.0+, there are some other dependencies below for building RAFT from source. Many of the dependencies are optional and depend only on the primitives being used. All of these can be installed with cmake or [rapids-cpm](https://github.com/rapidsai/rapids-cmake#cpm) and many of them can be installed with [conda](https://anaconda.org).

#### Required
- [RMM](https://github.com/rapidsai/rmm) corresponding to RAFT version.
- [Thrust](https://github.com/NVIDIA/thrust) v1.17 / [CUB](https://github.com/NVIDIA/cub)
- [cuCollections](https://github.com/NVIDIA/cuCollections) - Used in `raft::sparse::distance` API.
- [CUTLASS](https://github.com/NVIDIA/cutlass)  v2.9.1 - Used in `raft::distance` API.

#### Optional
- [NCCL](https://github.com/NVIDIA/nccl) - Used in `raft::comms` API and needed to build `raft-dask`.
- [UCX](https://github.com/openucx/ucx) - Used in `raft::comms` API and needed to build `raft-dask`.
- [Googletest](https://github.com/google/googletest) - Needed to build tests
- [Googlebench](https://github.com/google/benchmark) - Needed to build benchmarks
- [Doxygen](https://github.com/doxygen/doxygen) - Needed to build docs

All of RAFT's C++ APIs can be used header-only but pre-compiled shared libraries also contain some host-accessible APIs and template instantiations to accelerate compile times.

The recommended way to build and install RAFT is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python artifacts and provides options for building and installing the headers, tests, benchmarks, and individual shared libraries.

### Header-only C++

`build.sh` uses [rapids-cmake](https://github.com/rapidsai/rapids-cmake), which will automatically download any dependencies which are not already installed. It's important to note that while all the headers will be installed and available, some parts of the RAFT API depend on libraries like CUTLASS, which will need to be explicitly enabled in `build.sh`.

The following example will download the needed dependencies and install the RAFT headers into `$INSTALL_PREFIX/include/raft`. 
```bash
./build.sh libraft

```
The `-n` flag can be passed to just have the build download the needed dependencies. Since RAFT is primarily used at build-time, the dependencies will never be installed by the RAFT build.
```bash
./build.sh libraft -n
```

Once installed, `libraft` headers (and dependencies which were downloaded and installed using `rapids-cmake`) can be uninstalled also using `build.sh`:
```bash
./build.sh libraft --uninstall
```


### C++ Shared Library (optional)

A shared library can be built for speeding up compile times. The shared library also contains a runtime API that allows you to invoke RAFT APIs directly from C++ source files (without `nvcc`). The shared library can also significantly improve re-compile times both while developing RAFT and using its APIs to develop applications. Pass the `--compile-lib` flag to `build.sh` to build the library:
```bash
./build.sh libraft --compile-lib
```

In above example the shared library is installed by default into `$INSTALL_PREFIX/lib`. To disable this, pass `-n` flag.

Once installed, the shared library, headers (and any dependencies downloaded and installed via `rapids-cmake`) can be uninstalled using `build.sh`:
```bash
./build.sh libraft --uninstall
```


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
./build.sh libraft tests --compile-lib
```

The tests are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_TEST`.

For example, to run the distance tests:
```bash
./cpp/build/DISTANCE_TEST
```

It can take sometime to compile all of the tests. You can build individual tests by providing a semicolon-separated list to the `--limit-tests` option in `build.sh`:

```bash
./build.sh libraft tests -n --limit-tests=NEIGHBORS_TEST;DISTANCE_TEST;MATRIX_TEST
```

### Benchmarks

The benchmarks are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_BENCH`.
```bash
./build.sh libraft bench
```

It can take sometime to compile all of the benchmarks. You can build individual benchmarks by providing a semicolon-separated list to the `--limit-bench` option in `build.sh`:

```bash
./build.sh libraft bench -n --limit-bench=NEIGHBORS_BENCH;DISTANCE_BENCH;LINALG_BENCH
```

### C++ Using Cmake Directly

Use `CMAKE_INSTALL_PREFIX` to install RAFT into a specific location. The snippet below will install it into the current conda environment:
```bash
cd cpp
mkdir build
cd build
cmake -D BUILD_TESTS=ON -DRAFT_COMPILE_LIBRARY=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make -j<parallel_level> install
```

RAFT's cmake has the following configurable flags available:.

| Flag                      | Possible Values      | Default Value | Behavior |
|---------------------------|----------------------| --- | --- |
| BUILD_TESTS               | ON, OFF              | ON | Compile Googletests |
| BUILD_PRIMS_BENCH               | ON, OFF              | OFF | Compile benchmarks |
| BUILD_ANN_BENCH               | ON, OFF              | OFF | Compile end-to-end ANN benchmarks |
| raft_FIND_COMPONENTS      | compiled distributed | | Configures the optional components as a space-separated list |
| RAFT_COMPILE_LIBRARY      | ON, OFF              | ON if either BUILD_TESTS or BUILD_PRIMS_BENCH is ON; otherwise OFF | Compiles all `libraft` shared libraries (these are required for Googletests) |
| DETECT_CONDA_ENV          | ON, OFF              | ON | Enable detection of conda environment for dependencies |
| RAFT_NVTX                 | ON, OFF              | OFF | Enable NVTX Markers |
| CUDA_ENABLE_KERNELINFO    | ON, OFF              | OFF | Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer` |
| CUDA_ENABLE_LINEINFO      | ON, OFF              | OFF | Enable the -lineinfo option for nvcc |
| CUDA_STATIC_RUNTIME       | ON, OFF              | OFF | Statically link the CUDA runtime |

Currently, shared libraries are provided for the `libraft-nn` and `libraft-distance` components.

### Python

Conda environment scripts are provided for installing the necessary dependencies for building and using the Python APIs. It is preferred to use `mamba`, as it provides significant speedup over `conda`. In addition you will have to manually install `nvcc` as it will not be installed as part of the conda environment. The following example will install create and install dependencies for a CUDA 11.8 conda environment:

```bash
mamba env create --name raft_env_name -f conda/environments/all_cuda-118_arch-x86_64.yaml
mamba activate raft_env_name
```

The Python APIs can be built and installed using the `build.sh` script:

```bash
# to build pylibraft
./build.sh libraft pylibraft --compile-lib
# to build raft-dask
./build.sh libraft pylibraft raft-dask --compile-lib
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

The Python packages can also be uninstalled using the `build.sh` script:
```bash
./build.sh pylibraft raft-dask --uninstall
```

### Documentation

The documentation requires that the C++ headers and python packages have been built and installed.

The following will build the docs along with the C++ and Python packages:

```
./build.sh libraft pylibraft raft-dask docs --compile-lib
```


## Using RAFT in downstream projects

There are a few different strategies for including RAFT in downstream projects, depending on whether the [required build dependencies](#build-dependencies) have already been installed and are available on the `lib` and `include` paths.

Using cmake, you can enable CUDA support right in your project's declaration:
```cmake
project(YOUR_PROJECT VERSION 0.1 LANGUAGES CXX CUDA)
```

Please note that some additional compiler flags might need to be added when building against RAFT. For example, if you see an error like this `The experimental flag '--expt-relaxed-constexpr' can be used to allow this.`. The necessary flags can be set with cmake:
```cmake
target_compile_options(your_target_name PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
```

Further, it's important that the language level be set to at least C++ 17. This can be done with cmake:
```cmake
set_target_properties(your_target_name
PROPERTIES CXX_STANDARD                        17
           CXX_STANDARD_REQUIRED               ON
           CUDA_STANDARD                       17
           CUDA_STANDARD_REQUIRED              ON
           POSITION_INDEPENDENT_CODE           ON
           INTERFACE_POSITION_INDEPENDENT_CODE ON)
```


### C++ header-only integration (without cmake)

While not a highly suggested method for building against RAFT, when all of the needed [build dependencies](#build-dependencies) are already satisfied, RAFT can be integrated into downstream projects by cloning the repository and adding `cpp/include` from RAFT to the include path:
```cmake
set(RAFT_GIT_DIR ${CMAKE_CURRENT_BINARY_DIR}/raft CACHE STRING "Path to RAFT repo")
ExternalProject_Add(raft
  GIT_REPOSITORY    git@github.com:rapidsai/raft.git
  GIT_TAG           branch-23.04
  PREFIX            ${RAFT_GIT_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   "")
set(RAFT_INCLUDE_DIR ${RAFT_GIT_DIR}/raft/cpp/include CACHE STRING "RAFT include variable")
```
### C++ header-only integration (with cmake)


When using cmake, you can install RAFT headers into your environment with `./build.sh libraft`. 

If the RAFT headers have already been installed into your environment with cmake or through conda, such as by using the `build.sh` script, use `find_package(raft)` and the `raft::raft` target.

### Using C++ pre-compiled shared libraries

Use `find_package(raft COMPONENTS compiled distributed)` to enable the shared library and transitively pass dependencies through separate targets for each component. In this example, the `raft::compiled` and `raft::distributed` targets will be available for configuring linking paths in addition to `raft::raft`. These targets will also pass through any transitive dependencies (such as NCCL for the `distributed` component).

The pre-compiled libraries contain template specializations for commonly used types, such as single- and double-precision floating-point. In order to use the symbols in the pre-compiled libraries, the compiler needs to be told not to instantiate templates that are already contained in the shared libraries. By convention, these header files are named `specializations.cuh` and located in the base directory for the packages that contain specializations.

The following example tells the compiler to ignore the pre-compiled templates for the `raft::distance` API so any symbols already compiled into the `libraft` shared library will be used instead. RAFT's cmake creates a variable `RAFT_COMPILED` which can be used to ignore the pre-compiled template specializations only when the shared library has been enabled through cmake (such as by specifying the `compiled` component in `find_package`):
```c++
#ifdef RAFT_COMPILED
#include <raft/distance/distance.cuh>
#include <raft/distance/specializations.cuh>
#endif
```

### Building RAFT C++ from source in cmake

RAFT uses the [RAPIDS-CMake](https://github.com/rapidsai/rapids-cmake) library so it can be more easily included into downstream projects. RAPIDS cmake provides a convenience layer around the [CMake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake).

The following example is similar to invoking `find_package(raft)` but uses `rapids_cpm_find`, which provides a richer and more flexible configuration landscape by using CPM to fetch any dependencies not already available to the build. The `raft::raft` link target will be made available and it's recommended that it be used as a `PRIVATE` link dependency in downstream projects. The `COMPILE_LIBRARY` option enables the building the shared libraries.

The following `cmake` snippet enables a flexible configuration of RAFT:

```cmake

set(RAFT_VERSION "23.04")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAFT_VERSION}")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG COMPILE_LIBRARY)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN} )
  
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
          FIND_PACKAGE_ARGUMENTS "COMPONENTS compiled distributed"
          OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_PRIMS_BENCH OFF"
          "BUILD_ANN_BENCH OFF"
          "RAFT_COMPILE_LIBRARY ${PKG_COMPILE_LIBRARY}"
  )

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${RAFT_VERSION}.00
        FORK             ${RAFT_FORK}
        PINNED_TAG       ${RAFT_PINNED_TAG}
        COMPILE_LIBRARY          NO
)
```

You can find a fully-functioning [example template project](../../cpp/template/README.md) in the `cpp/template` directory, which provides everything you need to build a new application with RAFT or incorporate RAFT Into your existing libraries.

## Uninstall

Once built and installed, RAFT can be safely uninstalled using `build.sh` by specifying any or all of the installed components. Please note that since `pylibraft` depends on `libraft`, uninstalling `pylibraft` will also uninstall `libraft`:
```bash
./build.sh libraft pylibraft raft-dask --uninstall
```

Leaving off the installed components will uninstall everything that's been installed:
```bash
./build.sh --uninstall
```
