NVTX
====

This page provides C++ class references for the publicly-exposed elements of the `raft/core` package. The `raft/core` headers
require minimal dependencies, can be compiled without `nvcc`, and thus are safe to expose on your own public APIs. Aside from
the headers in the `raft/core` include directory, any headers in the codebase with the suffix `_types.hpp` are also safe to
expose in public APIs.

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/core/nvtx.hpp>``

namespace *raft::core*

.. doxygennamespace:: raft::common::nvtx
    :project: RAFT
    :members:
    :content-only:


