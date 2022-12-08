Matrix-Vector Operations
========================

This page provides C++ class references for the publicly-exposed elements of the `raft/linalg` (dense) linear algebra headers.
In addition to providing highly optimized arithmetic and matrix/vector operations, RAFT provides a consistent user experience
by providing common BLAS routines, standard linear system solvers, factorization and eigenvalue solvers. Some of these routines
hide the complexities of lower-level C-based libraries provided in the CUDA toolkit

.. role:: py(code)
   :language: c++
   :class: highlight

Arithmetic
----------

``#include <raft/linalg/matrix_vector.cuh>``

namespace *raft::linalg*

.. doxygengroup:: matrix_vector
    :project: RAFT
    :members:
    :content-only:


Operations
----------

``#include <raft/linalg/matrix_vector_op.cuh>``

namespace *raft::linalg*

.. doxygengroup:: matrix_vector_op
    :project: RAFT
    :members:
    :content-only:

