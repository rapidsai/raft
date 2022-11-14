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


Interruptible
#############

.. doxygenclass:: raft::interruptible
    :project: RAFT
    :members:

NVTX
####

.. doxygennamespace:: raft::common::nvtx
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

