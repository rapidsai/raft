~~~~~~~~~~~~~~~~~~~~~~~
PyLibRAFT API Reference
~~~~~~~~~~~~~~~~~~~~~~~

.. role:: py(code)
   :language: python
   :class: highlight


Approximate Nearest Neighbors
=============================

.. autoclass:: pylibraft.neighbors.ivf_pq.IndexParams

.. autofunction:: pylibraft.neighbors.ivf_pq.build

.. autofunction:: pylibraft.neighbors.ivf_pq.extend

..autoclass:: pylibraft.neighbors.ivf_pq.SearchParams

.. autofunction:: pylibraft.neighbors.ivf_pq.search


Cluster
=======

.. autofunction:: pylibraft.cluster.compute_new_centroids


Pairwise Distances
==================

.. autofunction:: pylibraft.distance.pairwise_distance

.. autofunction:: pylibraft.distance.fused_l2_nn_armin


Random
======

.. autofunction:: pylibraft.random.rmat
