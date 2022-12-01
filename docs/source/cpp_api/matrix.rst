Matrix
======

This page provides C++ class references for the publicly-exposed elements of the `raft/matrix` headers. The `raft/matrix`
headers cover many operations on matrices that are otherwise not covered by `raft/linalg`.

.. role:: py(code)
   :language: c++
   :class: highlight


Arithmetic
##########

Line-wise Operation
-------------------

``#include <raft/matrix/linewise_op.cuh>``

namespace *raft::matrix*

.. doxygengroup:: linewise_op
    :project: RAFT
    :members:
    :content-only:

Power
-----

``#include <raft/matrix/power.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_power
    :project: RAFT
    :members:
    :content-only:

Ratio
-----

``#include <raft/matrix/ratio.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_ratio
    :project: RAFT
    :members:
    :content-only:

Reciprocal
----------

``#include <raft/matrix/reciprocal.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_reciprocal
    :project: RAFT
    :members:
    :content-only:

Sign-flip
---------

``#include <raft/matrix/sign_flip.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_sign_flip
    :project: RAFT
    :members:
    :content-only:


Square Root
-----------

``#include <raft/matrix/sqrt.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_sqrt
    :project: RAFT
    :members:
    :content-only:


Initialization
##############

Init
----

``#include <raft/matrix/init.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_init
    :project: RAFT
    :members:
    :content-only:


Copying
#######

Copy
----

``#include <raft/matrix/copy.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_copy
    :project: RAFT
    :members:
    :content-only:

Gather
------

``#include <raft/matrix/gather.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_gather
    :project: RAFT
    :members:
    :content-only:



Slicing, Extraction, and Truncation
###################################

Argmax
------

``#include <raft/matrix/argmax.cuh>``

namespace *raft::matrix*

.. doxygengroup:: argmax
    :project: RAFT
    :members:
    :content-only:

Argmin
------

``#include <raft/matrix/argmin.cuh>``

namespace *raft::matrix*

.. doxygengroup:: argmin
    :project: RAFT
    :members:
    :content-only:

Diagonal
--------

``#include <raft/matrix/diagonal.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_diagonal
    :project: RAFT
    :members:
    :content-only:

Slicing
-------

``#include <raft/matrix/slice.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_slice
    :project: RAFT
    :members:
    :content-only:

Triangular
----------

``#include <raft/matrix/triangular.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_triangular
    :project: RAFT
    :members:
    :content-only:

Manipulation
############

Reverse
-------

``#include <raft/matrix/reverse.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_reverse
    :project: RAFT
    :members:
    :content-only:

Threshold
---------

``#include <raft/matrix/threshold.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_threshold
    :project: RAFT
    :members:
    :content-only:


Ordering
########

Column-wise Sort
----------------

``#include <raft/matrix/col_wise_sort.cuh>``

namespace *raft::matrix*

.. doxygengroup:: col_wise_sort
    :project: RAFT
    :members:
    :content-only:


Reduction
#########

Matrix Norm
-----------

``#include <raft/matrix/norm.cuh>``

namespace *raft::matrix*

.. doxygengroup:: matrix_norm
    :project: RAFT
    :members:
    :content-only:


.. doxygennamespace:: raft::matrix
    :project: RAFT
    :members:
    :content-only: