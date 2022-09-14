Spatial
=======

This page provides C++ class references for the publicly-exposed elements of the spatial package.

Distance
########

.. doxygennamespace:: raft::distance
    :project: RAFT


Nearest Neighbors
#################

.. doxygenfunction:: raft::spatial::knn::brute_force_knn
    :project: RAFT

.. doxygenfunction:: raft::spatial::knn::select_k
    :project: RAFT

.. doxygenfunction:: raft::spatial::knn::knn_merge_parts
    :project: RAFT


IVF-Flat
--------

.. doxygennamespace:: raft::spatial::knn::ivf_flat
    :project: RAFT
    :members:
