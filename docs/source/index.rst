RAPIDS RAFT: Reusable Accelerated Functions and Tools
=====================================================

RAFT contains fundamental widely-used algorithms and primitives for scientific computing, data science and machine learning. The algorithms are CUDA-accelerated and form building-blocks for rapidly composing analytics.

By taking a primitives-based approach to algorithm development, RAFT

- accelerates algorithm construction time
- reduces the maintenance burden by maximizing reuse across projects, and
- centralizes core reusable computations, allowing future optimizations to benefit all algorithms that use them.


While not exhaustive, the following general categories help summarize the accelerated building blocks that RAFT contains:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Category
     - Examples
   * - Data Formats
     - sparse & dense, conversions, data generation
   * - Dense Operations
     - linear algebra, matrix and vector operations, slicing, norms, factorization, least squares, svd & eigenvalue problems
   * - Sparse Operations
     - linear algebra, eigenvalue problems, slicing, norms, reductions, factorization, symmetrization, components & labeling
   * - Spatial
     - pairwise distances, nearest neighbors, neighborhood graph construction
   * - Basic Clustering
     - spectral clustering, hierarchical clustering, k-means
   * - Solvers
     - combinatorial optimization, iterative solvers
   * - Statistics
     - sampling, moments and summary statistics, metrics
   * - Tools & Utilities
     - common utilities for developing CUDA applications, multi-node multi-gpu infrastructure

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick_start.md
   build.md
   developer_guide.md
   cpp_api.rst
   pylibraft_api.rst
   raft_ann_benchmarks.md
   raft_dask_api.rst
   using_comms.rst
   using_libraft.md
   contributing.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
