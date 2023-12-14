CAGRA
=====

CAGRA is a graph-based nearest neighbors implementation with state-of-the art query performance for both small- and large-batch sized search.

Please note that the CAGRA implementation is currently experimental and the API is subject to change from release to release. We are currently working on promoting CAGRA to a top-level stable API within RAFT.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/neighbors/cagra.cuh>``

namespace *raft::neighbors::cagra*

.. doxygengroup:: cagra
    :project: RAFT
    :members:
    :content-only:


Serializer Methods
------------------
``#include <raft/neighbors/cagra_serialize.cuh>``

namespace *raft::neighbors::cagra*

.. doxygengroup:: cagra_serialize
    :project: RAFT
    :members:
    :content-only:

CAGRA index build and hnswlib search
------------------------------------
``#include <raft/neighbors/cagra_hnswlib.hpp>``

namespace *raft::neighbors::cagra_hnswlib*

.. doxygengroup:: cagra_hnswlib
    :project: RAFT
    :members:
    :content-only:
