# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Framework Toolkit</div>

RAFT contains fundamental widely-used algorithms and primitives for data science, graph and machine learning. The algorithms are CUDA-accelerated and form building-blocks for rapidly composing analytics in the [RAPIDS](https://rapids.ai) ecosystem. 

By taking a primitives-based approach to algorithm development, RAFT
- accelerates algorithm construction time
- reduces the maintenance burden by maximizing reuse across projects, and
- centralizes the core computations, allowing future optimizations to benefit all algorithms that use them.

The algorithms in RAFT span the following general categories:
#####
| Category | Examples |
| --- | --- |
| **Data Formats** | sparse & dense, conversions, data generation |
| **Data Generation** | sparse, spatial, machine learning datasets |
| **Dense Linear Algebra** | matrix arithmetic, norms, factorization, least squares, svd & eigenvalue problems |
| **Spatial** | pairwise distances, nearest neighbors, neighborhood graph construction |
| **Sparse Operations** | linear algebra, eigenvalue problems, slicing, symmetrization, labeling |
| **Basic Clustering** | spectral clustering, hierarchical clustering, k-means |
| **Optimization** | combinatorial optimization, iterative solvers |
| **Statistics** | sampling, moments and summary statistics, metrics |
| **Distributed Tools** | multi-node multi-gpu infrastructure |

RAFT provides a header-only C++ library and pre-compiled shared libraries that can 1) speed up compile times and 2) enable the APIs to be used without CUDA-enabled compilers.

RAFT also provides a Python library that is currently limited to
1. a python wrapper around the `raft::handle_t` for managing cuda library resources
2. definitions for using `raft::handle_t` directly in cython
3. tools for building multi-node multi-GPU algorithms that leverage [Dask](https://dask.org/)

The Python API is being improved to wrap the algorithms and primitives from the categories above.

## Getting started

### Rapids Memory Manager (RMM)
RAFT relies heavily on [RMM](https://github.com/rapidsai/rmm) which, 
like other projects in the RAPIDS ecosystem, eases the burden of configuring different allocation strategies globally 
across the libraries that use it. RMM also provides [RAII](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)) wrappers around device arrays that handle the allocation and cleanup.

### C++ Example

Most of the primitives in RAFT accept a `raft::handle_t` object for the management of resources which are expensive to create, such CUDA streams, stream pools, and handles to other CUDA libraries like `cublas` and `cusolver`.

The example below demonstrates creating a RAFT handle and using it with RMM's `device_uvector` to allocate memory on device and compute
pairwise Euclidean distances:
```c++
#include <raft/handle.hpp>
#include <raft/distance/distance.hpp>

#include <rmm/device_uvector.hpp>
raft::handle_t handle;

int n_samples = ...;
int n_features = ...;

rmm::device_uvector<float> input(n_samples * n_features, handle.get_stream());
rmm::device_uvector<float> output(n_samples * n_samples, handle.get_stream());

// ... Populate feature matrix ...

auto metric = raft::distance::DistanceType::L2SqrtExpanded;
rmm::device_uvector<char> workspace(0, handle.get_stream());
raft::distance::pairwise_distance(handle, input.data(), input.data(),
                                  output.data(),
                                  n_samples, n_samples, n_features,
                                  workspace.data(), metric);
```

## Installing

RAFT can be installed through conda, cmake-package-manager (cpm), or by building the repository from source. 

### Conda

The easiest way to install RAFT is through conda and several packages are provided.
- `libraft-runtime` contains a subset of CUDA/C++ headers that can be safely included in public APIs because they depend only upon the cudatoolkit libraries and can be safely compiled without `nvcc`
- `libraft-nn` (optional) contains shared libraries for the nearest neighbors primitives.
- `libraft-distance` (optional) contains shared libraries for distance primitives.
- `pyraft` (optional) contains reusable Python tools to accelerate Python algorithm development

To install RAFT with conda (change to `rapidsai-nightly` for more up-to-date but less stable nightly packages)
```bash
conda install -c rapidsai libraft-runtime libraft-nn libraft-distance pyraft
```

After installing RAFT, `find_package(raft COMPONENTS nn distance)` can be used in your CUDA/C++ build. Note that the `COMPONENTS` are optional and will depend on the packages installed.

### CPM

RAFT uses the [RAPIDS cmake](https://github.com/rapidsai/rapids-cmake) library, which makes it simple to include in downstream cmake projects. RAPIDS cmake provides a convenience layer around the [Cmake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake). 

After [installing](https://github.com/rapidsai/rapids-cmake#installation) rapids-cmake in your project, you can begin using RAFT by placing the code snippet below in a file named `get_raft.cmake` and including it in your cmake build with `include(get_raft.cmake)`. This will create the `raft::raft` target to add to configure the link libraries for your artifacts.

```cmake

set(RAFT_VERSION "22.04")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG USE_FAISS_STATIC 
          COMPILE_LIBRARIES ENABLE_NN_DEPENDENCIES)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

  #-----------------------------------------------------
  # Invoke CPM find_package()
  #-----------------------------------------------------

  rapids_cpm_find(raft ${PKG_VERSION}
          GLOBAL_TARGETS      raft::raft
          BUILD_EXPORT_SET    proj-exports
          INSTALL_EXPORT_SET  proj-exports
          CPM_ARGS
          GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
          GIT_TAG        ${PKG_PINNED_TAG}
          SOURCE_SUBDIR  cpp
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
find_and_configure_raft(VERSION    ${RAFT_VERSION}.00
        FORK             rapidsai
        PINNED_TAG       branch-${RAFT_VERSION}

        COMPILE_LIBRARIES      NO
        ENABLE_NN_DEPENDENCIES NO
        USE_FAISS_STATIC       NO
)
```

### Source

The easiest way to build RAFT from source is to use the `build.sh` script at the root of the repository:
1. Create an environment with the needed dependencies: `conda env create --name raft_dev -f conda/environments/raft_dev_cuda11.5.yml`
2. Run the build script from the repository root: `./build.sh pyraft libraft --compile-libs`

The [Build](BUILD.md) instructions contain more details on building RAFT from source and including it in downstream projects. You can find a more comprehensive version of the above CPM code snippet the [Building RAFT C++ from source](BUILD.md#build_cxx_source) guide.

## Folder Structure and Contents

The folder structure mirrors other RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `ci`: Scripts for running CI in PRs
- `conda`: Conda recipes and development conda environments
- `cpp`: Source code for all C++ code. 
  - `docs`: Doxygen configuration
  - `include`: The C++ API is fully-contained here
  - `src`: Compiled template specializations for the shared libraries
- `docs`: Source code and scripts for building library documentation (doxygen + pydocs)
- `python`: Source code for all Python source code.

## Contributing

If you are interested in contributing to the RAFT project, please read our [Contributing guidelines](CONTRIBUTING.md). Refer to the [Developer Guide](DEVELOPER_GUIDE.md) for details on the developer guidelines, workflows, and principals. 
