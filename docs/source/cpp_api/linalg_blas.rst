BLAS Routines
=============

This page provides C++ class references for the publicly-exposed elements of the `raft/linalg` (dense) linear algebra headers.
In addition to providing highly optimized arithmetic and matrix/vector operations, RAFT provides a consistent user experience
by providing common BLAS routines, standard linear system solvers, factorization and eigenvalue solvers. Some of these routines
hide the complexities of lower-level C-based libraries provided in the CUDA toolkit

.. role:: py(code)
   :language: c++
   :class: highlight

axpy
----

``#include <raft/linalg/axpy.cuh>``

namespace *raft::linalg*

.. doxygengroup:: axpy
    :project: RAFT
    :members:
    :content-only:

dot
---

``#include <raft/linalg/dot.cuh>``

namespace *raft::linalg*

.. doxygengroup:: dot
    :project: RAFT
    :members:
    :content-only:

gemm
----

``#include <raft/linalg/gemm.cuh>``

namespace *raft::linalg*

.. doxygengroup:: gemm
    :project: RAFT
    :members:
    :content-only:

gemv
----

``#include <raft/linalg/gemv.cuh>``

namespace *raft::linalg*

.. doxygengroup:: gemv
    :project: RAFT
    :members:
    :content-only:

