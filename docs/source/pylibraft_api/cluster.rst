Cluster
=======

This page provides pylibraft class references for the publicly-exposed elements of the `pylibraft.cluster` package.

.. note::

  The vector search and clustering algorithms in RAFT are being migrated to a new library dedicated to vector search called `cuVS <https://github.com/rapidsai/cuvs>`_. We will continue to support the vector search algorithms in RAFT during this move, will no longer update them after the RAPIDS 24.06 (June) release. We plan to complete the migration by RAPIDS 24.08 (August) release.


.. role:: py(code)
   :language: python
   :class: highlight

KMeans
######

.. autoclass:: pylibraft.cluster.kmeans.KMeansParams
    :members:

.. autofunction:: pylibraft.cluster.kmeans.fit

.. autofunction:: pylibraft.cluster.kmeans.cluster_cost

.. autofunction:: pylibraft.cluster.kmeans.compute_new_centroids
