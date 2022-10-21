Core
====

This page provides C++ class references for the publicly-exposed elements of the `raft/core` package. The `raft/core` headers
require minimal dependencies, can be compiled without `nvcc`, and thus are safe to expose on your own public APIs. Aside from
the headers in the `raft/core` include directory, any headers in the codebase with the suffix `_types.hpp` are also safe to
expose in public APIs.

handle_t
########

.. doxygenclass:: raft::handle_t
    :project: RAFT
    :members:


interruptible
#############

.. doxygenclass:: raft::interruptible
    :project: RAFT
    :members:

NVTX
####

.. doxygennamespace:: raft::common::nvtx
    :project: RAFT
    :members:


mdarray
#######

.. doxygenclass:: raft::mdarray
    :project: RAFT
    :members:

.. doxygenclass:: raft::make_device_matrix
    :project: RAFT

.. doxygenclass:: raft::make_device_vector
    :project: RAFT

.. doxygenclass:: raft::make_device_scalar
    :project: RAFT

.. doxygenclass:: raft::make_host_matrix
    :project: RAFT

.. doxygenclass:: raft::make_host_vector
    :project: RAFT

.. doxygenclass:: raft::make_device_scalar
    :project: RAFT


mdspan
#######

.. doxygenfunction:: raft::make_device_mdspan
    :project: RAFT

.. doxygenfunction:: raft::make_device_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
    :project: RAFT

.. doxygenfunction:: raft::make_host_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
    :project: RAFT

Device Factories
----------------

.. doxygenfunction:: raft::make_device_matrix
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar
    :project: RAFT

Host Factories
----------------

.. doxygenfunction:: raft::make_host_matrix
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar
    :project: RAFT


mdspan
#######

Device Vocabulary
-----------------

.. doxygentypedef:: raft::device_mdspan
   :project: RAFT

.. doxygentypedef:: raft::device_matrix_view
   :project: RAFT

.. doxygentypedef:: raft::device_vector_view
   :project: RAFT

.. doxygentypedef:: raft::device_scalar_view
   :project: RAFT

Host Vocabulary
---------------

.. doxygentypedef:: raft::host_mdspan
   :project: RAFT

.. doxygentypedef:: raft::host_matrix_view
   :project: RAFT

.. doxygentypedef:: raft::host_vector_view
   :project: RAFT

.. doxygentypedef:: raft::host_scalar_view
   :project: RAFT

Device Factories
----------------

.. doxygenfunction:: raft::make_device_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
    :project: RAFT

Host Factories
--------------

.. doxygenfunction:: raft::make_host_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
    :project: RAFT

span
####

.. doxygentypedef:: raft::device_span
   :project: RAFT

.. doxygentypedef:: raft::host_span
   :project: RAFT

.. doxygenclass:: raft::span
    :project: RAFT
    :members:



Key-Value Pair
##############

.. doxygenstruct:: raft::KeyValuePair
    :project: RAFT
    :members:


logger
######

.. doxygenclass:: raft::logger
    :project: RAFT
    :members:


Multi-node Multi-GPU
####################

.. doxygennamespace:: raft::comms
    :project: RAFT
    :members:

