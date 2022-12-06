span: One-dimensional Non-owning View
=====================================

This page provides C++ class references for the RAFT's 1d span and multi-dimension owning (mdarray) and non-owning (mdspan) APIs. These headers can be found in the `raft/core` directory.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/core/span.hpp>``

.. doxygenclass:: raft::span
    :project: RAFT
    :members:

``#include <raft/core/device_span.hpp>``

.. doxygentypedef:: raft::device_span
   :project: RAFT

``#include <raft/core/host_span.hpp>``

.. doxygentypedef:: raft::host_span
   :project: RAFT

