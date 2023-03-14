IVF-PQ
======

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/neighbors/ivf_pq.cuh>``

namespace *raft::neighbors::ivf_pq*

.. doxygengroup:: ivf_pq
    :project: RAFT
    :members:
    :content-only:

Serializer Methods
------------------
``#include <raft/neighbors/ivf_pq_serialize.cuh>``

.. doxygenfunction:: serialize(raft::device_resources const& handle, std::ostream& os, const index<IdxT>& index)
    :project: RAFT

.. doxygenfunction:: serialize(raft::device_resources const& handle, const std::string& filename, const index<IdxT>& index)
    :project: RAFT

.. doxygenfunction:: deserialize(raft::device_resources const& handle, std::istream& is)
    :project: RAFT

.. doxygenfunction:: deserialize(raft::device_resources const& handle, const std::string& filename)
    :project: RAFT