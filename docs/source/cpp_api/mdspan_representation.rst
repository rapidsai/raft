Multi-dimensional Representation
================================

.. role:: py(code)
   :language: c++
   :class: highlight

Data Layouts
-------------

``#include <raft/core/mdspan.hpp>``

.. doxygentypedef:: raft::row_major
    :project: RAFT

.. doxygentypedef:: raft::col_major
    :project: RAFT


Shapes
------

``#include <raft/core/mdspan.hpp>``

.. doxygentypedef:: raft::matrix_extent
    :project: RAFT

.. doxygentypedef:: raft::vector_extent
    :project: RAFT

.. doxygentypedef:: raft::scalar_extent
    :project: RAFT

.. doxygentypedef:: raft::extent_3d
    :project: RAFT

.. doxygentypedef:: raft::extent_4d
    :project: RAFT

.. doxygentypedef:: raft::extent_5d
    :project: RAFT

.. doxygengroup:: mdspan_reshape
    :project: RAFT
    :members:
    :content-only:

.. doxygengroup:: mdarray_reshape
    :project: RAFT
    :members:
    :content-only:


Accessors
---------

``#include <raft/core/host_device_accessor.hpp>``

.. doxygenstruct:: raft::host_device_accessor
    :project: RAFT
    :members:

.. doxygentypedef:: raft::host_accessor
    :project: RAFT

.. doxygentypedef:: raft::device_accessor
    :project: RAFT

.. doxygentypedef:: raft::managed_accessor
    :project: RAFT


