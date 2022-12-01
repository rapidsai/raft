Core
====

This page provides C++ class references for the publicly-exposed elements of the `raft/core` package. The `raft/core` headers
require minimal dependencies, can be compiled without `nvcc`, and thus are safe to expose on your own public APIs. Aside from
the headers in the `raft/core` include directory, any headers in the codebase with the suffix `_types.hpp` are also safe to
expose in public APIs.

.. role:: py(code)
   :language: c++
   :class: highlight


handle_t
########

#include <raft/core/handle.hpp>

.. doxygenclass:: raft::handle_t
    :project: RAFT
    :members:


Interruptible
#############

``#include <raft/core/interupptible.hpp>``

.. doxygenclass:: raft::interruptible
    :project: RAFT
    :members:


NVTX
####

``#include <raft/core/nvtx.hpp>``

.. doxygennamespace:: raft::common::nvtx
    :project: RAFT
    :members:
    :content-only:


Key-Value Pair
##############

``#include <raft/core/kvp.hpp>``

.. doxygenstruct:: raft::KeyValuePair
    :project: RAFT
    :members:


logger
######

``#include <raft/core/logger.hpp>``

.. doxygenclass:: raft::logger
    :project: RAFT
    :members:


Multi-node Multi-GPU
####################

``#include <raft/core/comms.hpp>``

.. doxygennamespace:: raft::comms
    :project: RAFT
    :members:
    :content-only:

