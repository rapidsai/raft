mdarray: Multi-dimensional Owning Container
===========================================

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/core/mdarray.hpp>``

.. doxygenclass:: raft::mdarray
    :project: RAFT
    :members:

.. doxygenclass:: raft::array_interface
    :project: RAFT
    :members:

.. doxygenstruct:: raft::is_array_interface
    :project: RAFT
    :members:

.. doxygentypedef:: raft::is_array_interface_t
    :project RAFT

Device Vocabulary
-----------------

``#include <raft/core/device_mdarray.hpp>``

.. doxygentypedef:: raft::device_mdarray
    :project: RAFT

.. doxygentypedef:: raft::device_matrix
    :project: RAFT

.. doxygentypedef:: raft::device_vector
    :project: RAFT

.. doxygentypedef:: raft::device_scalar
    :project: RAFT


Device Factories
----------------

``#include <raft/core/device_mdarray.hpp>``

.. doxygenfunction:: raft::make_device_matrix
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar
    :project: RAFT


Host Vocabulary
---------------

``#include <raft/core/host_mdarray.hpp>``

.. doxygentypedef:: raft::host_matrix
    :project: RAFT

.. doxygentypedef:: raft::host_vector
    :project: RAFT

.. doxygentypedef:: raft::host_scalar
    :project: RAFT


Host Factories
--------------

``#include <raft/core/host_mdarray.hpp>``

.. doxygenfunction:: raft::make_host_matrix
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar
    :project: RAFT
