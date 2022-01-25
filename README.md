# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Framework Toolkit</div>

RAFT is a library containing building-blocks for rapid composition of RAPIDS Analytics. These building-blocks include shared representations, mathematical computational primitives, and utilities that accelerate building analytics and data science algorithms in the RAPIDS ecosystem. Both the C++ and Python components can be included in consuming libraries, providing operations for both dense and sparse matrix formats in the following general categories:

#####
| Category | Description / Examples |
| --- | --- |
| **Data Formats** | tensor representations and conversions for both sparse and dense formats |
| **Data Generation** | graph, spatial, and machine learning dataset generation |
| **Dense Operations** | linear algebra, statistics |
| **Spatial** | pairwise distances, nearest neighbors, neighborhood / proximity graph construction |
| **Sparse/Graph Operations** | linear algebra, statistics, slicing, msf, spectral embedding/clustering, slhc, vertex degree |
| **Solvers** | eigenvalue decomposition, least squares, lanczos |
| **Tools** | multi-node multi-gpu communicator, utilities |

By taking a primitives-based approach to algorithm development, RAFT accelerates algorithm construction time and reduces
the maintenance burden by maximizing reuse across projects. RAFT relies on the [RAPIDS memory manager (RMM)](https://github.com/rapidsai/rmm) which, 
like other projects in the RAPIDS ecosystem, eases the burden of configuring different allocation strategies globally 
across the libraries that use it. RMM also provides RAII wrappers around device arrays that handle the allocation and cleanup.

RAFT's primary goals are to be fast, simple, reusable, composable, and comprehensive.

## Build/Install RAFT

Refer to the [Build](BUILD.md) instructions for details on building and including the RAFT library in downstream projects.

## Getting started

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

## Folder Structure and Contents

The folder structure mirrors other RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `ci`: Scripts for running CI in PRs
- `conda`: conda recipes and development conda environments
- `cpp`: Source code for all C++ code. The code is currently header-only, therefore it is in the `include` folder (with no `src`).
- `docs`: Source code and scripts for building library documentation
- `python`: Source code for all Python source code.

## Contributing

If you are interested in contributing to the RAFT project, please read our [Contributing guidelines](CONTRIBUTING.md). Refer to the [Developer Guide](DEVELOPER_GUIDE.md) for details on the developer guidelines, workflows, and principals. 
