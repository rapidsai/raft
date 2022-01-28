# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Framework Toolkit</div>

RAFT contains fundamental widely-used algorithms and primitives for data science and ML. The algorithms are CUDA-accelerated and form building-blocks for rapidly composing analytics in the [RAPIDS](https://rapids.ai) ecosystem. 

By taking a primitives-based approach to algorithm development, RAFT
1. accelerates algorithm construction time
2. reduces the maintenance burden by maximizing reuse across projects, and
3. centralizes the core computations, allowing future optimizations to benefit all algorithms that use them.

RAFT provides a header-only C++ API with optional shared libraries that contain algorithms in the following general categories:

#####
| Category | Description / Examples |
| --- | --- |
| **Data Formats** | sparse & dense, conversions, and data generations |
| **Data Generation** | sparse, spatial, machine learning datasets |
| **Dense Linear Algebra** | matrix arithmetic, norms, factorization, least squares & eigenvalue problems |
| **Spatial** | pairwise distances, nearest neighbors, neighborhood graph construction |
| **Sparse Operations** | linear algebra, slicing, symmetrization, norms, spectral embedding, msf |
| **Basic Clustering** | spectral clustering, hierarchical clustering, k-means |
| **Iterative Solvers** | lanczos algorithm |
| **Statistics** | sampling, moments, metrics |
| **Distributed Tools** | multi-node multi-gpu infrastructure |

RAFT also provides a Python API that enables the building of multi-node multi-GPU algorithms in the [Dask](https://dask.org/) ecosystem. We are continuing to improve the coverage of the Python API to expose the building-blocks from the categories above.

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

## Build/Install RAFT

Refer to the [Build](BUILD.md) instructions for details on building and including the RAFT library in downstream projects.

## Folder Structure and Contents

The folder structure mirrors other RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `ci`: Scripts for running CI in PRs
- `conda`: Conda recipes and development conda environments
- `cpp`: Source code for all C++ code. 
  - `include`: The C++ API is fully-contained here 
  - `src`: Compiled template specializations for the shared libraries
- `docs`: Source code and scripts for building library documentation
- `python`: Source code for all Python source code.

## Contributing

If you are interested in contributing to the RAFT project, please read our [Contributing guidelines](CONTRIBUTING.md). Refer to the [Developer Guide](DEVELOPER_GUIDE.md) for details on the developer guidelines, workflows, and principals. 
