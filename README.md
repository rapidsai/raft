# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: Reusable Accelerated Functions and Tools</div>

RAFT contains fundamental widely-used algorithms and primitives for data science and machine learning. The algorithms are CUDA-accelerated and form building-blocks for rapidly composing analytics.

By taking a primitives-based approach to algorithm development, RAFT 
- accelerates algorithm construction time
- reduces the maintenance burden by maximizing reuse across projects, and
- centralizes core reusable computations, allowing future optimizations to benefit all algorithms that use them.

While not exhaustive, the following general categories help summarize the accelerated functions in RAFT:
#####
| Category | Examples |
| --- | --- |
| **Data Formats** | sparse & dense, conversions, data generation |
| **Dense Linear Algebra** | matrix arithmetic, norms, factorization, least squares, svd & eigenvalue problems |
| **Spatial** | pairwise distances, nearest neighbors, neighborhood graph construction |
| **Sparse Operations** | linear algebra, eigenvalue problems, slicing, symmetrization, labeling |
| **Basic Clustering** | spectral clustering, hierarchical clustering, k-means |
| **Solvers** | combinatorial optimization, iterative solvers |
| **Statistics** | sampling, moments and summary statistics, metrics |
| **Distributed Tools** | multi-node multi-gpu infrastructure |

RAFT provides a header-only C++ library and pre-compiled shared libraries that can 1) speed up compile times and 2) enable the APIs to be used without CUDA-enabled compilers.

RAFT also provides 2 Python libraries:
- `pylibraft` - low-level Python wrappers around RAFT algorithms and primitives.
- `raft-dask` - reusable infrastructure for building analytics, including tools for building both single-GPU and multi-node multi-GPU algorithms.

## Getting started

### RAPIDS Memory Manager (RMM)

RAFT relies heavily on RMM which eases the burden of configuring different allocation strategies globally across the libraries that use it.

### Multi-dimensional Arrays

The APIs in RAFT currently accept raw pointers to device memory and we are in the process of simplifying the APIs with the [mdspan](https://arxiv.org/abs/2010.06474) multi-dimensional array view for representing data in higher dimensions similar to the `ndarray` in the Numpy Python library. RAFT also contains the corresponding owning `mdarray` structure, which simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory. 

The `mdarray` forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:

```c++
#include <raft/mdarray.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_device_scalar<float>(handle, 1.0);
auto vector = raft::make_device_vector<float>(handle, n_cols);
auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);
```

### C++ Example

Most of the primitives in RAFT accept a `raft::handle_t` object for the management of resources which are expensive to create, such CUDA streams, stream pools, and handles to other CUDA libraries like `cublas` and `cusolver`.

The example below demonstrates creating a RAFT handle and using it with `device_matrix` and `device_vector` to allocate memory, generating random clusters, and computing
pairwise Euclidean distances:
```c++
#include <raft/handle.hpp>
#include <raft/mdarray.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/distance/distance.cuh>

raft::handle_t handle;

int n_samples = 5000;
int n_features = 50;

auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
auto labels = raft::make_device_vector<int>(handle, n_samples);
auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);

raft::random::make_blobs(handle, input.view(), labels.view());

auto metric = raft::distance::DistanceType::L2SqrtExpanded;
raft::distance::pairwise_distance(handle, input.view(), input.view(), output.view(), metric);
```

### Python Example

The `pylibraft` package contains a Python API for RAFT algorithms and primitives. The package is currently limited to pairwise distances, and we will continue adding more.

The example below demonstrates computing the pairwise Euclidean distances between cupy arrays. `pylibraft` is a low-level API that prioritizes efficiency and simplicity over being pythonic, which is shown here by pre-allocating the output memory before invoking the `pairwise_distance` function.

```python
import cupy as cp

from pylibraft.distance import pairwise_distance

n_samples = 5000
n_features = 50

in1 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
in2 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
output = cp.empty((n_samples, n_samples), dtype=cp.float32)

pairwise_distance(in1, in2, output, metric="euclidean")
```

## Installing

RAFT itself can be installed through conda, [Cmake Package Manager (CPM)](https://github.com/cpm-cmake/CPM.cmake), or by building the repository from source. Please refer to the [build instructions](BUILD.md) for more a comprehensive guide on building RAFT and using it in downstream projects.

### Conda

The easiest way to install RAFT is through conda and several packages are provided.
- `libraft-headers` RAFT headers
- `libraft-nn` (optional) contains shared libraries for the nearest neighbors primitives.
- `libraft-distance` (optional) contains shared libraries for distance primitives.
- `pylibraft` (optional) Python wrappers around RAFT algorithms and primitives
- `raft-dask` (optional) enables deployment of multi-node multi-GPU algorithms that use RAFT `raft::comms` in Dask clusters.

Use the following command to install RAFT with conda (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command.
```bash
mamba install -c rapidsai libraft-headers libraft-nn libraft-distance raft-dask pylibraft
```

After installing RAFT, `find_package(raft COMPONENTS nn distance)` can be used in your CUDA/C++ cmake build to compile and/or link against needed dependencies in your raft target. `COMPONENTS` are optional and will depend on the packages installed.

### CPM

RAFT uses the [RAPIDS-CMake](https://github.com/rapidsai/rapids-cmake) library, which makes it simple to include in downstream cmake projects. RAPIDS CMake provides a convenience layer around CPM. 

After [installing](https://github.com/rapidsai/rapids-cmake#installation) rapids-cmake in your project, you can begin using RAFT by placing the code snippet below in a file named `get_raft.cmake` and including it in your cmake build with `include(get_raft.cmake)`. This will make available several targets to add to configure the link libraries for your artifacts.

```cmake

set(RAFT_VERSION "22.04")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAFT_VERSION}")

function(find_and_configure_raft)
  set(oneValueArgs VERSION FORK PINNED_TAG COMPILE_LIBRARIES)
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
          OPTIONS
          "BUILD_TESTS OFF"
          "BUILD_BENCH OFF"
          "RAFT_COMPILE_LIBRARIES ${PKG_COMPILE_LIBRARIES}"
  )

endfunction()

# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION    ${RAFT_VERSION}.00
        FORK             ${RAFT_FORK}
        PINNED_TAG       ${RAFT_PINNED_TAG}
        COMPILE_LIBRARIES      NO
)
```

Several CMake targets can be made available by adding components in the table below to the `RAFT_COMPONENTS` list above, separated by spaces. The `raft::raft` target will always be available. RAFT headers require, at a minimum, the CUDA toolkit libraries and RMM dependencies.

| Component | Target | Description | Base Dependencies |
| --- | --- | --- | --- |
| n/a | `raft::raft` | Full RAFT header library | CUDA toolkit library, RMM, Thrust (optional), NVTools (optional) |
| distance | `raft::distance` | Pre-compiled template specializations for raft::distance | raft::raft, cuCollections (optional)  |
| nn | `raft::nn` | Pre-compiled template specializations for raft::spatial::knn | raft::raft, FAISS (optional) |

### Source

The easiest way to build RAFT from source is to use the `build.sh` script at the root of the repository:
1. Create an environment with the needed dependencies: 
```
mamba env create --name raft_dev_env -f conda/environments/raft_dev_cuda11.5.yml
mamba activate raft_dev_env
```
```
./build.sh raft-dask pylibraft libraft tests bench --compile-libs
```

The [build](BUILD.md) instructions contain more details on building RAFT from source and including it in downstream projects. You can also find a more comprehensive version of the above CPM code snippet the [Building RAFT C++ from source](BUILD.md#build_cxx_source) section of the build instructions.

## Folder Structure and Contents

The folder structure mirrors other RAPIDS repos, with the following folders:

- `ci`: Scripts for running CI in PRs
- `conda`: Conda recipes and development conda environments
- `cpp`: Source code for C++ libraries. 
  - `docs`: Doxygen configuration
  - `include`: The C++ API is fully-contained here
  - `src`: Compiled template specializations for the shared libraries
- `docs`: Source code and scripts for building library documentation (doxygen + pydocs)
- `python`: Source code for Python libraries.

## Contributing

If you are interested in contributing to the RAFT project, please read our [Contributing guidelines](CONTRIBUTING.md). Refer to the [Developer Guide](DEVELOPER_GUIDE.md) for details on the developer guidelines, workflows, and principals. 

## References

When citing RAFT generally, please consider referencing this Github project.
```bibtex
@misc{rapidsai, 
  title={Rapidsai/raft: RAFT contains fundamental widely-used algorithms and primitives for data science, Graph and machine learning.},
  url={https://github.com/rapidsai/raft}, 
  journal={GitHub}, 
  publisher={Nvidia RAPIDS}, 
  author={Rapidsai},
  year={2022}
}
```
If citing the sparse pairwise distances API, please consider using the following bibtex:
```bibtex
@article{nolet2021semiring,
  title={Semiring primitives for sparse neighborhood methods on the gpu},
  author={Nolet, Corey J and Gala, Divye and Raff, Edward and Eaton, Joe and Rees, Brad and Zedlewski, John and Oates, Tim},
  journal={arXiv preprint arXiv:2104.06357},
  year={2021}
}
```
