Core
====

This page provides C++ class references for the publicly-exposed elements of the `raft/core` package. The `raft/core` headers
require minimal dependencies, can be compiled without `nvcc`, and thus are safe to expose on your own public APIs. Aside from
the headers in the `raft/core` include directory, any headers in the codebase with the suffix `_types.hpp` are also safe to
expose in public APIs.

.. role:: py(code)
   :language: c++
   :class: highlight

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core_resources.rst
   core_logger.rst
   core_kvp.rst
   core_nvtx.rst
   core_interruptible.rst
   core_operators.rst
   core_math.rst