Neighbors
=========

This page provides pylibraft class references for the publicly-exposed elements of the neighbors package.

.. role:: py(code)
   :language: python
   :class: highlight


Brute Force
###########

.. autofunction:: pylibraft.neighbors.brute_force.knn


CAGRA
#####

.. autoclass:: pylibraft.neighbors.cagra.IndexParams
    :members:

.. autofunction:: pylibraft.neighbors.cagra.build

.. autoclass:: pylibraft.neighbors.cagra.SearchParams
    :members:

.. autofunction:: pylibraft.neighbors.cagra.search

Serializer Methods
------------------
.. autofunction:: pylibraft.neighbors.cagra.save

.. autofunction:: pylibraft.neighbors.cagra.load

CAGRA hnswlib
#############

.. autoclass:: pylibraft.neighbors.cagra_hnswlib.SearchParams
    :members:

.. autofunction:: pylibraft.neighbors.cagra_hnswlib.search

Serializer Methods
------------------
.. autofunction:: pylibraft.neighbors.cagra_hnswlib.save

.. autofunction:: pylibraft.neighbors.cagra_hnswlib.load

IVF-Flat
########

.. autoclass:: pylibraft.neighbors.ivf_flat.IndexParams
    :members:

.. autofunction:: pylibraft.neighbors.ivf_flat.build

.. autofunction:: pylibraft.neighbors.ivf_flat.extend

.. autoclass:: pylibraft.neighbors.ivf_flat.SearchParams
    :members:

.. autofunction:: pylibraft.neighbors.ivf_flat.search

Serializer Methods
------------------

.. autofunction:: pylibraft.neighbors.ivf_flat.save

.. autofunction:: pylibraft.neighbors.ivf_flat.load

IVF-PQ
######

.. autoclass:: pylibraft.neighbors.ivf_pq.IndexParams
    :members:

.. autofunction:: pylibraft.neighbors.ivf_pq.build

.. autofunction:: pylibraft.neighbors.ivf_pq.extend

.. autoclass:: pylibraft.neighbors.ivf_pq.SearchParams
    :members:

.. autofunction:: pylibraft.neighbors.ivf_pq.search

Serializer Methods
------------------

.. autofunction:: pylibraft.neighbors.ivf_pq.save

.. autofunction:: pylibraft.neighbors.ivf_pq.load

Candidate Refinement
--------------------

.. autofunction:: pylibraft.neighbors.refine
