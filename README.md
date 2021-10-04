# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Frameworks Toolkit</div>

RAFT is a header-only library of C++ primitives for building analytics and data science algorithms in the RAPIDS ecosystem. RAFT primitives operate on both dense and sparse matrix formats in the following categories:
#####
| Category | Description | Location(s) |
| --- | --- | -- |
| **Formats and conversion** | sparse / dense tensor representations and conversions | `sparse/`, `sparse/convert`, `matrix/` |
| **Data generation** | graph, spatial, and machine learning dataset generation | `random/`, TBD |
| **Matrix operations** | sparse / dense matrix arithmetic and reductions | `linalg/`, `matrix/`, `sparse/linalg`, `sparse/op`|
| **Graph algorithms** | clustering, layout, components analysis, spanning trees | `sparse/mst`, `sparse/hierarchy`, `sparse/linalg`, `spectral/` |
| **Spatial algorithms** | pairwise distances, nearest neighbors, neighborhood / proximity graph construction | `distance/`, `spatial/knn`, `sparse/selection`, `sparse/distance` |
| **Solvers** | linear solvers such as eigenvalue decomposition, svd, and lanczos | `linalg/`, `sparse/linalg` |
| **Distributed GPU analytics** | synchronous communications abstraction layer (CAL) and Python integration w/ Dask | `comms/` |

The library layout contains the following structure:
```bash
cpp/include/raft
     |------------ comms      [communication abstraction layer]
     |------------ distance   [dense pairwise distances]
     |------------ linalg     [dense linear algebra]
     |------------ matrix     [dense matrix format]
     |------------ random     [random matrix generation]
     |------------ sparse     [sparse matrix and graph algorithms]
     |------------ spatial    [spatial algorithms]
     |------------ spectral   [spectral clustering]
     |------------ stats      [statistics primitives]
     |------------ handle.hpp [raft handle]
```

By taking a primitives-based approach to algorithm development, RAFT accelerates algorithm construction time and reduces
the maintenance burden by maximizing reuse, even across projects. RAFT relies on the [RAPIDS memory manager (RMM)](https://github.com/rapidsai/rmm) 
like other projects in the RAPIDS ecosystem. RMM eases the burden of configuring different allocation strategies globally 
across the libraries that use it. RMM also provides RAII wrappers around device arrays that handle the allocation and cleanup.

The example below demonstrates using RMM's `device_uvector` to allocate memory on device and using RAFT to compute
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

Refer to the [Build and Development Guide](BUILD.md) for details on RAFT's design, building, testing and development guidelines.

## Folder Structure and Contents

The folder structure mirrors the main RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `cpp`: Source code for all C++ code. The code is header only, therefore it is in the `include` folder (with no `src`).
- `python`: Source code for all Python source code.
- `ci`: Scripts for running CI in PRs


