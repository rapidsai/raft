Linear Algebra Solvers
======================

This page provides C++ class references for the publicly-exposed elements of the `raft/linalg` (dense) linear algebra headers.
In addition to providing highly optimized arithmetic and matrix/vector operations, RAFT provides a consistent user experience
by providing common BLAS routines, standard linear system solvers, factorization and eigenvalue solvers. Some of these routines
hide the complexities of lower-level C-based libraries provided in the CUDA toolkit

.. role:: py(code)
   :language: c++
   :class: highlight

Eigen Decomposition
-------------------

``#include <raft/linalg/eig.cuh>``

namespace *raft::linalg*

.. doxygengroup:: eig
    :project: RAFT
    :members:
    :content-only:

QR Decomposition
----------------

``#include <raft/linalg/qr.cuh>``

namespace *raft::linalg*

.. doxygengroup:: qr
    :project: RAFT
    :members:
    :content-only:

Randomized Singular-Value Decomposition
---------------------------------------

``#include <raft/linalg/rsvd.cuh>``

namespace *raft::linalg*

.. doxygengroup:: rsvd
    :project: RAFT
    :members:
    :content-only:

Singular-Value Decomposition
----------------------------

``#include <raft/linalg/svd.cuh>``

namespace *raft::linalg*

.. doxygengroup:: svd
    :project: RAFT
    :members:
    :content-only:

Least Squares
-------------

``#include <raft/linalg/lstsq.cuh>``

namespace *raft::linalg*

.. doxygengroup:: lstsq
    :project: RAFT
    :members:
    :content-only:
