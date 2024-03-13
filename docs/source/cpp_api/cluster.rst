Cluster
=======

This page provides C++ API references for the publicly-exposed elements of the `raft/cluster` headers. RAFT provides
fundamental clustering algorithms which are, themselves, considered reusable building blocks for other algorithms.

.. note::

  The vector search and clustering algorithms in RAFT are being migrated to a new library dedicated to vector search called `cuVS <https://github.com/rapidsai/cuvs>`_. We will continue to support the vector search algorithms in RAFT during this move, will no longer update them after the RAPIDS 24.06 (June) release. We plan to complete the migration by RAPIDS 24.08 (August) release.


.. role:: py(code)
   :language: c++
   :class: highlight

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   cluster_kmeans.rst
   cluster_kmeans_balanced.rst
   cluster_slhc.rst
   cluster_spectral.rst