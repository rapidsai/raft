Linear Algebra
==============

This page provides C++ class references for the publicly-exposed elements of the `raft/linalg` (dense) linear algebra headers.
In addition to providing highly optimized arithmetic and matrix/vector operations, RAFT provides a consistent user experience
by providing common BLAS routines, standard linear system solvers, factorization and eigenvalue solvers. Some of these routines
hide the complexities of lower-level C-based libraries provided in the CUDA toolkit 

.. role:: py(code)
   :language: c++
   :class: highlight


Element-wise Arithmetic
#######################

Addition
--------

``#include <raft/linalg/add.cuh>``

namespace *raft::linalg*

.. doxygengroup:: add_dense
    :project: RAFT
    :members:
    :content-only:


Binary Op
---------

``#include <raft/linalg/binary_op.cuh>``

namespace *raft::linalg*

.. doxygengroup:: binary_op
    :project: RAFT
    :members:
    :content-only:

Division
--------

``#include <raft/linalg/divide.cuh>``

namespace *raft::linalg*

.. doxygengroup:: divide
    :project: RAFT
    :members:
    :content-only:

Multiplication
--------------

``#include <raft/linalg/multiply.cuh>``

namespace *raft::linalg*

.. doxygengroup:: multiply
    :project: RAFT
    :members:
    :content-only:

Power
-----

``#include <raft/linalg/power.cuh>``

namespace *raft::linalg*

.. doxygengroup:: power
    :project: RAFT
    :members:
    :content-only:

Square Root
-----------

``#include <raft/linalg/sqrt.cuh>``

namespace *raft::linalg*

.. doxygengroup:: sqrt
    :project: RAFT
    :members:
    :content-only:

Subtraction
-----------

``#include <raft/linalg/subtract.cuh>``

namespace *raft::linalg*

.. doxygengroup:: sub
    :project: RAFT
    :members:
    :content-only:

Ternary Op
----------

``#include <raft/linalg/ternary_op.cuh>``

namespace *raft::linalg*

.. doxygengroup:: ternary_op
    :project: RAFT
    :members:
    :content-only:

Unary Op
--------

``#include <raft/linalg/unary_op.cuh>``

namespace *raft::linalg*

.. doxygengroup:: unary_op
    :project: RAFT
    :members:
    :content-only:

Mapping, Reductions, and Norms
##############################

Coalesced Reduction
-------------------

``#include <raft/linalg/coalesced_reduction.cuh>``

namespace *raft::linalg*

.. doxygengroup:: coalesced_reduction
    :project: RAFT
    :members:
    :content-only:

Map
---

``#include <raft/linalg/map.cuh>``

namespace *raft::linalg*

.. doxygengroup:: map
    :project: RAFT
    :members:
    :content-only:

Map Reduce
----------

``#include <raft/linalg/map_reduce.cuh>``

namespace *raft::linalg*

.. doxygengroup:: map_reduce
    :project: RAFT
    :members:
    :content-only:

Mean Squared Error
------------------


``#include <raft/linalg/mean_squared_error.cuh>``

namespace *raft::linalg*

.. doxygengroup:: mean_squared_error
    :project: RAFT
    :members:
    :content-only:

Norm
----

``#include <raft/linalg/norm.cuh>``

namespace *raft::linalg*

.. doxygengroup:: norm
    :project: RAFT
    :members:
    :content-only:

Normalize
---------

``#include <raft/linalg/normalize.cuh>``

namespace *raft::linalg*

.. doxygengroup:: normalize
    :project: RAFT
    :members:
    :content-only:

Reduction
---------

``#include <raft/linalg/reduce.cuh>``

namespace *raft::linalg*

.. doxygengroup:: reduction
    :project: RAFT
    :members:
    :content-only:

Reduce Cols By Key
------------------

``#include <raft/linalg/reduce_cols_by_key.cuh>``

namespace *raft::linalg*

.. doxygengroup:: reduce_cols_by_key
    :project: RAFT
    :members:
    :content-only:

Reduce Rows By Key
------------------

``#include <raft/linalg/reduce_rows_by_key.cuh>``

namespace *raft::linalg*

.. doxygengroup:: reduce_rows_by_key
    :project: RAFT
    :members:
    :content-only:

Strided Reduction
-----------------

``#include <raft/linalg/strided_reduction.cuh>``

namespace *raft::linalg*

.. doxygengroup:: strided_reduction
    :project: RAFT
    :members:
    :content-only:


Matrix
######

Transpose
---------

``#include <raft/linalg/transpose.cuh>``

namespace *raft::linalg*

.. doxygengroup:: transpose
    :project: RAFT
    :members:
    :content-only:



Matrix-Vector
#############

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


BLAS Routines
#############

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

Solvers
#######

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
