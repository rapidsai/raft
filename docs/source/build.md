# Installation

RAFT currently provides libraries for C++ and Python. The C++ libraries, including the header-only and optional shared library, can be installed with Conda. 

Both the C++ and Python APIs require CMake to build from source.

## Table of Contents

- [Install C++ and Python through Conda](#installing-c-and-python-through-conda)
- [Installing Python through Pip](#installing-python-through-pip)
- [Building C++ and Python from source](#building-c-and-python-from-source)
  - [CUDA/GPU requirements](#cudagpu-requirements)
  - [Build dependencies](#build-dependencies)
    - [Required](#required)
    - [Optional](#optional)
    - [Conda environment scripts](#conda-environment-scripts)
  - [Header-only C++](#header-only-c)
  - [C++ shared library](#c-shared-library-optional)
  - [ccache and sccache](#ccache-and-sccache)
  - [C++ tests](#c-tests)
  - [C++ primitives microbenchmarks](#c-primitives-microbenchmarks)
  - [Python libraries](#python-libraries)
- [Using CMake directly](#using-cmake-directly)
- [Build documentation](#build-documentation)
- [Using RAFT in downstream projects](#using-raft-c-in-downstream-projects)
  - [CMake targets](#cmake-targets)

------

## Installing C++ and Python through Conda

The easiest way to install RAFT is through conda and several packages are provided.
- `libraft-headers` C++ headers
- `libraft` (optional) C++ shared library containing pre-compiled template instantiations and runtime API.
- `pylibraft` (optional) Python library
- `raft-dask` (optional) Python library for deployment of multi-node multi-GPU algorithms that use the RAFT `raft::comms` abstraction layer in Dask clusters.
- `raft-ann-bench` (optional) Benchmarking tool for easily producing benchmarks that compare RAFT's vector search algorithms against other state-of-the-art implementations.
- `raft-ann-bench-cpu` (optional) Reproducible benchmarking tool similar to above, but doesn't require CUDA to be installed on the machine. Can be used to test in environments with competitive CPUs.

Use the following command, depending on your CUDA version, to install all of the RAFT packages with conda (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command.
```bash
# for CUDA 11.8
mamba install -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft cuda-version=11.8
```

```bash
# for CUDA 12.0
mamba install -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft cuda-version=12.0
```

Note that the above commands will also install `libraft-headers` and `libraft`.

You can also install the conda packages individually using the `mamba` command above. For example, if you'd like to install RAFT's headers and pre-compiled shared library to use in your project:
```bash
# for CUDA 12.0
mamba install -c rapidsai -c conda-forge -c nvidia libraft libraft-headers cuda-version=12.0
```

If installing the C++ APIs Please see [using libraft](https://docs.rapids.ai/api/raft/nightly/using_libraft/) for more information on using the pre-compiled shared library. You can also refer to the [example C++ template project](https://github.com/rapidsai/raft/tree/branch-23.12/cpp/template) for a ready-to-go CMake configuration that you can drop into your project and build against installed RAFT development artifacts above.

## Installing Python through Pip

`pylibraft` and `raft-dask` both have packages that can be [installed through pip](https://rapids.ai/pip.html#install). 

For CUDA 11 packages:
```bash
pip install pylibraft-cu11 --extra-index-url=https://pypi.nvidia.com
pip install raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
```

And CUDA 12 packages:
```bash
pip install pylibraft-cu12 --extra-index-url=https://pypi.nvidia.com
pip install raft-dask-cu12 --extra-index-url=https://pypi.nvidia.com
```

These packages statically build RAFT's pre-compiled instantiations, so the C++ headers and pre-compiled shared library won't be readily available to use in your code. 

## Building C++ and Python from source

### CUDA/GPU Requirements
- cmake 3.26.4+
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

#### Conda environment scripts

Conda environment scripts are provided for installing the necessary dependencies to build both the C++ and Python libraries from source. It is preferred to use `mamba`, as it provides significant speedup over `conda`:
```bash
mamba env create --name rapids_raft -f conda/environments/all_cuda-120_arch-x86_64.yaml
mamba activate rapids_raft
```

All of RAFT's C++ APIs can be used header-only and optional pre-compiled shared libraries provide some host-accessible runtime APIs and template instantiations to accelerate compile times.

The process for building from source with CUDA 11 differs slightly in that your host system will also need to have CUDA toolkit installed which is greater than, or equal to, the version you install into you conda environment. Installing CUDA toolkit into your host system is necessary because `nvcc` is not provided with Conda's cudatoolkit dependencies for CUDA 11. The following example will install create and install dependencies for a CUDA 11.8 conda environment
```bash
mamba env create --name rapids_raft -f conda/environments/all_cuda-118_arch-x86_64.yaml
mamba activate rapids_raft
```

The recommended way to build and install RAFT from source is to use the `build.sh` script in the root of the repository. This script can build both the C++ and Python artifacts and provides CMake options for building and installing the headers, tests, benchmarks, and the pre-compiled shared library.

### Header-only C++

`build.sh` uses [rapids-cmake](https://github.com/rapidsai/rapids-cmake), which will automatically download any dependencies which are not already installed. It's important to note that while all the headers will be installed and available, some parts of the RAFT API depend on libraries like CUTLASS, which will need to be explicitly enabled in `build.sh`.

The following example will download the needed dependencies and install the RAFT headers into `$INSTALL_PREFIX/include/raft`. 
```bash
./build.sh libraft
```
The `-n` flag can be passed to just have the build download the needed dependencies. Since RAFT's C++ headers are primarily used during build-time in downstream projects, the dependencies will never be installed by the RAFT build.
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

### C++ Tests

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

### C++ Primitives Microbenchmarks

The benchmarks are broken apart by algorithm category, so you will find several binaries in `cpp/build/` named `*_PRIMS_BENCH`.
```bash
./build.sh libraft bench-prims
```

It can take sometime to compile all of the benchmarks. You can build individual benchmarks by providing a semicolon-separated list to the `--limit-bench-prims` option in `build.sh`:

```bash
./build.sh libraft bench-prims -n --limit-bench=NEIGHBORS_PRIMS_BENCH;DISTANCE_PRIMS_BENCH;LINALG_PRIMS_BENCH
```

In addition to microbenchmarks for individual primitives, RAFT contains a reproducible benchmarking tool for evaluating the performance of RAFT's vector search algorithms against the existing state-of-the-art. Please refer to the [RAFT ANN Benchmarks](https://docs.rapids.ai/api/raft/nightly/raft_ann_benchmarks/) guide for more information on this tool.

### Python libraries

The Python libraries can be built and installed using the `build.sh` script:

```bash
# to build pylibraft
./build.sh libraft pylibraft --compile-lib
# to build raft-dask (depends on pylibraft)
./build.sh libraft pylibraft raft-dask --compile-lib
```

`setup.py` can also be used to build the Python libraries manually:

```bash
cd python/raft-dask
python setup.py build_ext --inplace
python setup.py install

cd python/pylibraft
python setup.py build_ext --inplace
python setup.py install
```

Python tests are automatically installed with the corresponding libraries. To run Python tests:
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

### Using CMake directly

When building RAFT from source, the `build.sh` script offers a nice wrapper around the `cmake` commands to ease the burdens of manually configuring the various available cmake options. When more fine-grained control over the CMake configuration is desired, the `cmake` command can be invoked directly as the below example demonstrates. 

The `CMAKE_INSTALL_PREFIX` installs RAFT into a specific location. The example below installs RAFT into the current Conda environment:
```bash
cd cpp
mkdir build
cd build
cmake -D BUILD_TESTS=ON -DRAFT_COMPILE_LIBRARY=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ../
make -j<parallel_level> install
```

RAFT's CMake has the following configurable flags available:

| Flag                            | Possible Values      | Default Value | Behavior                                                                     |
|---------------------------------|----------------------| --- |------------------------------------------------------------------------------|
| BUILD_TESTS                     | ON, OFF              | ON | Compile Googletests                                                          |
| BUILD_PRIMS_BENCH               | ON, OFF              | OFF | Compile benchmarks                                                           |
| BUILD_ANN_BENCH                 | ON, OFF              | OFF | Compile end-to-end ANN benchmarks |
| CUDA_ENABLE_KERNELINFO          | ON, OFF              | OFF | Enables `kernelinfo` in nvcc. This is useful for `compute-sanitizer`         |
| CUDA_ENABLE_LINEINFO            | ON, OFF              | OFF | Enable the -lineinfo option for nvcc                                         |
| CUDA_STATIC_RUNTIME             | ON, OFF              | OFF | Statically link the CUDA runtime                                             |
| DETECT_CONDA_ENV                | ON, OFF              | ON | Enable detection of conda environment for dependencies                       |
| raft_FIND_COMPONENTS            | compiled distributed | | Configures the optional components as a space-separated list                 |
| RAFT_COMPILE_LIBRARY            | ON, OFF              | ON if either BUILD_TESTS or BUILD_PRIMS_BENCH is ON; otherwise OFF | Compiles all `libraft` shared libraries (these are required for Googletests) |
| RAFT_ENABLE_CUBLAS_DEPENDENCY   | ON, OFF | ON | Link against cublas library in `raft::raft`                                  | 
| RAFT_ENABLE_CUSOLVER_DEPENDENCY | ON, OFF | ON | Link against cusolver library in `raft::raft`                                | 
| RAFT_ENABLE_CUSPARSE_DEPENDENCY | ON, OFF | ON | Link against cusparse library in `raft::raft`                                | 
| RAFT_ENABLE_CUSOLVER_DEPENDENCY | ON, OFF | ON | Link against curand library in `raft::raft`                                  | 
| RAFT_NVTX                       | ON, OFF              | OFF | Enable NVTX Markers                                                          |

### Build documentation

The documentation requires that the C++ and Python libraries have been built and installed. The following will build the docs along with the C++ and Python packages:

```
./build.sh libraft pylibraft raft-dask docs --compile-lib
```

## Using RAFT C++ in downstream projects

There are a few different strategies for including RAFT in downstream projects, depending on whether the [required build dependencies](#build-dependencies) have already been installed and are available on the `lib` and `include` search paths.

When using the GPU parts of RAFT, you will need to enable CUDA support in your CMake project declaration:
```cmake
project(YOUR_PROJECT VERSION 0.1 LANGUAGES CXX CUDA)
```

Note that some additional compiler flags might need to be added when building against RAFT. For example, if you see an error like this `The experimental flag '--expt-relaxed-constexpr' can be used to allow this.`. The necessary flags can be set with CMake:
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

The [C++ example template project](https://github.com/rapidsai/raft/tree/HEAD/cpp/template) provides an end-to-end buildable example of what a `CMakeLists.txt` that uses RAFT should look like. The items below point out some of the needed details.

#### CMake Targets

The `raft::raft` CMake target is made available when including RAFT into your CMake project but additional CMake targets can be made available by adding to the `COMPONENTS` option in CMake's `find_package(raft)` (refer to [CMake docs](https://cmake.org/cmake/help/latest/command/find_package.html#basic-signature) to learn more). The components should be separated by spaces. The `raft::raft` target will always be available. Note that the `distributed` component also exports additional dependencies.

| Component   | Target              | Description                                              | Base Dependencies                      |
|-------------|---------------------|----------------------------------------------------------|----------------------------------------|
| n/a         | `raft::raft`        | Full RAFT header library                                 | CUDA toolkit, RMM, NVTX, CCCL, CUTLASS |
| compiled    | `raft::compiled`    | Pre-compiled template instantiations and runtime library | raft::raft                             |
| distributed | `raft::distributed` | Dependencies for `raft::comms` APIs                      | raft::raft, UCX, NCCL         