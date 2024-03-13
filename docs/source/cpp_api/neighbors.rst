Neighbors
=========

This page provides C++ class references for the publicly-exposed elements of the neighbors package.

.. note::

  The vector search and clustering algorithms in RAFT are being migrated to a new library dedicated to vector search called `cuVS <https://github.com/rapidsai/cuvs>`_. We will continue to support the vector search algorithms in RAFT during this move, will no longer update them after the RAPIDS 24.06 (June) release. We plan to complete the migration by RAPIDS 24.08 (August) release.


.. role:: py(code)
   :language: c++
   :class: highlight

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   neighbors_brute_force.rst
   neighbors_ivf_flat.rst
   neighbors_ivf_pq.rst
   neighbors_epsilon_neighborhood.rst
   neighbors_ball_cover.rst
   neighbors_cagra.rst