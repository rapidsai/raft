# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Framework Toolkit</div>

RAFT contains fundamental widely-used algorithms and primitives for data science, graph and machine learning. The algorithms are CUDA-accelerated and form building-blocks for rapidly composing analytics in the [RAPIDS](https://rapids.ai) ecosystem. 

By taking a primitives-based approach to algorithm development, RAFT
1. accelerates algorithm construction time
2. reduces the maintenance burden by maximizing reuse across projects, and
3. centralizes the core computations, allowing future optimizations to benefit all algorithms that use them.

At its core, RAFT is a header-only C++ library with optional shared libraries that span the following categories:

#####
| Category | Examples |
| --- | --- |
| **Data Formats** | sparse & dense, conversions, data generation |
| **Data Generation** | sparse, spatial, machine learning datasets |
| **Dense Linear Algebra** | matrix arithmetic, norms, factorization, least squares, svd & eigenvalue problems |
| **Spatial** | pairwise distances, nearest neighbors, neighborhood graph construction |
| **Sparse Operations** | linear algebra, eigenvalue problems, slicing, symmetrization, connected component labeling |
| **Basic Clustering** | spectral clustering, hierarchical clustering, k-means |
| **Combinatorial Optimization** | linear assignment problem, minimum spanning forest |
| **Iterative Solvers** | lanczos |
| **Statistics** | sampling, moments and summary statistics, metrics |
| **Distributed Tools** | multi-node multi-gpu infrastructure |

RAFT also provides a Python library that includes
1. a python wrapper around the `raft::handle_t` for managing cuda library resources
2. building multi-node multi-GPU algorithms that leverage [Dask](https://dask.org/)

We are continuing to improve the Python API by exposing the core algorithms and primitives from the categories above.

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

## Build / Install RAFT

### Conda

RAFT has several packages that can be installed with conda:
- `libraft-headers` contains all the CUDA/C++ headers
- `libraft-nn` (optional) contains precompiled shared libraries for the nearest neighbors algorithms. If FAISS is not already installed in your environment, this will need to be installed to use the nearest neighbors headers.
- `libraft-distance` (optional) contains shared libraries for distance algorithms.
- `pyraft` (optional) contains the Python library

To install the RAFT nightly build
```bash
conda install -c rapidsai-nightly libraft-headers libraft-nn libraft-distance pyraft
```

After installing raft, you can add `find_package(raft COMPONENTS nn, distance)` to begin using it in your CUDA/C++ build. Note that the `COMPONENTS` are optional and will depend on the packages installed.


### CPM

RAFT uses the [RAPIDS cmake](https://github.com/rapidsai/rapids-cmake) library, so it can be easily included into downstream projects. RAPIDS cmake provides a convenience layer around the [Cmake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake). The following example is similar to building RAFT itself from source but allows it to be done in cmake, providing the `raft::raft` link target and `RAFT_INCLUDE_DIR` for includes. The `COMPILE_LIBRARIES` option enables the building of the shared libraries

```cmake

set(RAFT_VERSION "22.04")

function(find_and_configure_raft)

  set(oneValueArgs VERSION FORK PINNED_TAG USE_FAISS_STATIC 
          COMPILE_LIBRARIES ENABLE_NN_DEPENDENCIES CLONE_ON_PIN)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                            "${multiValueArgs}" ${ARGN} )

  if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${RAFT_VERSION}")
    message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
    execute_process(
            COMMAND git clone "https://github.com/${PKG_FORK}/raft.git" --branch ${PKG_PINNED_TAG} raft-source
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_deps)
    set(CPM_raft_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/_deps/raft-source)
  endif()


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
find_and_configure_raft(VERSION    ${RAFT_VERSION}.00
        FORK             rapidsai
        PINNED_TAG       branch-${RAFT_VERSION}

        # When PINNED_TAG above doesn't match cuml,
        # force local raft clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ON

        COMPILE_LIBRARIES      NO
        ENABLE_NN_DEPENDENCIES NO
        USE_FAISS_STATIC       NO
)
```

To use the above cmake code in your build, create a file `get_raft.cmake` in your project and include it in your cmake build with `include(get_raft.cmake)`. 

Refer to the [Build](BUILD.md) instructions for more details on building RAFT from source and including it in downstream projects.

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
