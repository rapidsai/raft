Distance
========

This page provides C++ class references for the publicly-exposed elements of the `raft/distance` package. RAFT's
distances have been highly optimized and support a wide assortment of different distance measures.

.. role:: py(code)
   :language: c++
   :class: highlight

Distance Types
##############

#include <raft/distance/distance_types.hpp>

.. doxygenenum:: raft::distance::DistanceType
   :project: RAFT


Pairwise Distance
#################

#include <raft/distance/distance.cuh>

.. doxygengroup:: distance_mdspan
    :project: RAFT
    :members:
    :content-only:


Fused 1-Nearest Neighbors
#########################

#include <raft/distance/fused_l2_nn.cuh>

.. doxygengroup:: fused_l2_nn
    :project: RAFT
    :members:
    :content-only:

