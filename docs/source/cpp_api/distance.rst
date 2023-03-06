Distance
========

This page provides C++ class references for the publicly-exposed elements of the `raft/distance` package. RAFT's
distances have been highly optimized and support a wide assortment of different distance measures.

.. role:: py(code)
   :language: c++
   :class: highlight

Distance Types
--------------

``#include <raft/distance/distance_types.hpp>``

namespace *raft::distance*

.. doxygenenum:: raft::distance::DistanceType
   :project: RAFT


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   distance_pairwise.rst
   distance_1nn.rst
   distance_masked_nn.rst
