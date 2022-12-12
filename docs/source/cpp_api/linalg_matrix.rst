Matrix Operations
=================

This page provides C++ class references for the publicly-exposed elements of the `raft/linalg` (dense) linear algebra headers.
In addition to providing highly optimized arithmetic and matrix/vector operations, RAFT provides a consistent user experience
by providing common BLAS routines, standard linear system solvers, factorization and eigenvalue solvers. Some of these routines
hide the complexities of lower-level C-based libraries provided in the CUDA toolkit

.. role:: py(code)
   :language: c++
   :class: highlight

Transpose
---------

``#include <raft/linalg/transpose.cuh>``

namespace *raft::linalg*

.. doxygengroup:: transpose
    :project: RAFT
    :members:
    :content-only:

