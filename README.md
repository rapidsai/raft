# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Frameworks Toolkit</div>

RAFT is a library containing shared formats, primitives, and utilities that accelerate building analytics and data science algorithms in the RAPIDS ecosystem. Both the C++ and Python components can be included in consuming libraries. RAFT primitives operate on both dense and sparse matrix formats, with building blocks spanning the following categories:
#####
| Category | Description |
| --- | --- |
| **Data Formats** | tensor representations and conversions for both sparse and dense formats |
| **Data Generation** | graph, spatial, and machine learning dataset generation |
| **Dense Matrix Operations** | dense matrix arithmetic and reductions |
| **Sparse Matrix Operations** | sparse matrix arithmetic and reductions, graph primitives |
| **Spatial Primitives** | pairwise distances, nearest neighbors, neighborhood / proximity graph construction |
| **Solvers** | eigenvalue decomposition, svd, lanczos |
| **Communicator** | UCX/NCCL communications abstraction and Python integration w/ Dask |

By taking a primitives-based approach to algorithm development, RAFT accelerates algorithm construction time and reduces
the maintenance burden by maximizing reuse across projects. RAFT relies on the [RAPIDS memory manager (RMM)](https://github.com/rapidsai/rmm) which, 
like other projects in the RAPIDS ecosystem, eases the burden of configuring different allocation strategies globally 
across the libraries that use it. RMM also provides RAII wrappers around device arrays that handle the allocation and cleanup.

## Getting started

Refer to the [Build and Development Guide](BUILD.md) for details on RAFT's design, building, testing and development guidelines.

Most of the primitives in RAFT accept a `raft::handle_t` object for the management of resources which are expensive to create, such CUDA streams, stream pools, and handles to other CUDA libraries like `cublas` and `cusolver`. 


### C++ Example

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

The folder structure mirrors the main RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `cpp`: Source code for all C++ code. The code is header only, therefore it is in the `include` folder (with no `src`).
- `python`: Source code for all Python source code.
- `ci`: Scripts for running CI in PRs

The C++ portion of RAFT is header-only and contains the following include directories:
```bash
cpp/include/raft
     |
     |------------ comms      [communications abstraction layer for distributed primitives]
     |
     |------------ distance   [dense pairwise distance primitives]
     |
     |------------ linalg     [dense linear algebra primitives]
     |
     |------------ matrix     [dense matrix format]
     |
     |------------ random     [random matrix generation]
     |
     |------------ sparse     [sparse matrix / graph primitives]
     |
     |------------ spatial    [spatial primitives] (might require FAISS)
     |
     |------------ spectral   [spectral clustering]
     |
     |------------ stats      [statistics primitives]
     |
     |------------ handle.hpp [raft handle]
```


