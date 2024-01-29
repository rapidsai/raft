HNSW
=====

HNSW is a graph-based nearest neighbors implementation for the CPU. 
This implementation provides the ability to serialize a CAGRA graph and read it as a base-layer-only hnswlib graph.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/neighbors/hnsw.hpp>``

namespace *raft::neighbors::hnsw*

.. doxygengroup:: hnsw
    :project: RAFT
    :members:
    :content-only:

Serializer Methods
------------------
``#include <raft/neighbors/hnsw_serialize.cuh>``

namespace *raft::neighbors::hnsw*

.. doxygengroup:: hnsw_serialize
    :project: RAFT
    :members:
    :content-only:
