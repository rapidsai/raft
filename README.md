# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: RAPIDS Analytics Frameworks Toolkit</div>

RAFT is a library of C++ primitives for building analytics and data science algorithms in the RAPIDS ecosystem. RAFT primitives operate on both dense and sparse matrix formats in the following categories:

##### 
| Category | Description |
| --- | --- |
| **Formats and conversion** | sparse / dense tensor representations and conversions |
| **Dataset generation** | graph, spatial, and machine learning dataset generation |
| **Matrix operations** | sparse / dense matrix arithmetic and reductions |
| **Graph algorithms** | clustering, layout, components analysis, spanning trees |
| **Neighborhoods and spatial distances** | distances, nearest neighbors, neighborhood / proximity graph construction |
| **Solvers** | linear solvers such as eigenvalue decomposition, svd, and lanczos |
| **Distributed GPU analytics** | communicator abstraction layer (CAL), CAL Python integration w/ Dask |

Refer to the [Build and Development Guide](BUILD.md) for details on RAFT's design, building, testing and development guidelines.

## Folder Structure and Contents

The folder structure mirrors the main RAPIDS repos (cuDF, cuML, cuGraph...), with the following folders:

- `cpp`: Source code for all C++ code. The code is header only, therefore it is in the `include` folder (with no `src`).
- `python`: Source code for all Python source code.
- `ci`: Scripts for running CI in PRs


