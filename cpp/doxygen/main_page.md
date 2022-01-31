# libraft

RAFT (RAPIDS Analytics Framework Toolkit) is a library containing building-blocks for rapid composition of RAPIDS Analytics. These building-blocks include shared representations, mathematical computational primitives, and utilities that accelerate building analytics and data science algorithms in the RAPIDS ecosystem. Both the C++ and Python components can be included in consuming libraries, providing building-blocks for both dense and sparse matrix formats in the following general categories:

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
