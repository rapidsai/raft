Core
====

This page provides C++ class references for the publicly-exposed elements of the core package.


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

span
####

.. doxygenclass:: raft::device_span
    :project: RAFT
    :members:

.. doxygenclass:: raft::host_span
    :project: RAFT
    :members:

Key-Value Pair
##############

.. doxygenclass:: raft::KeyValuePair
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

