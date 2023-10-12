mdspan: Multi-dimensional Non-owning View
==========================================

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/core/mdspan.hpp>``

.. doxygentypedef:: raft::mdspan
    :project: RAFT

.. doxygenfunction:: raft::make_mdspan
    :project: RAFT

.. doxygenfunction:: raft::make_extents
    :project: RAFT

.. doxygenfunction:: raft::make_strided_layout(Extents extents, Strides strides)
    :project: RAFT

.. doxygengroup:: mdspan_unravel
    :project: RAFT
    :members:
    :content-only:

.. doxygengroup:: mdspan_contiguous
    :project: RAFT
    :members:
    :content-only:

.. doxygengroup:: mdspan_make_const
    :project: RAFT
    :members:
    :content-only:


Device Vocabulary
-----------------

``#include <raft/core/device_mdspan.hpp>``

.. doxygentypedef:: raft::device_mdspan
   :project: RAFT

.. doxygenstruct:: raft::is_device_mdspan
   :project: RAFT

.. doxygentypedef:: raft::is_device_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::is_input_device_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::is_output_device_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::enable_if_device_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_input_device_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_output_device_mdspan
    :project: RAFT

.. doxygentypedef:: raft::device_matrix_view
   :project: RAFT

.. doxygentypedef:: raft::device_vector_view
   :project: RAFT

.. doxygentypedef:: raft::device_scalar_view
   :project: RAFT


Device Factories
----------------

``#include <raft/core/device_mdspan.hpp>``

.. doxygenfunction:: raft::make_device_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector_view(ElementType* ptr, IndexType n)
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
   :project: RAFT


Managed Vocabulary
------------------

``#include <raft/core/device_mdspan.hpp>``

..doxygentypedef:: raft::managed_mdspan
  :project: RAFT

.. doxygenstruct:: raft::is_managed_mdspan
   :project: RAFT

.. doxygentypedef:: raft::is_managed_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::is_input_managed_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::is_output_managed_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::enable_if_managed_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_input_managed_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_output_managed_mdspan
    :project: RAFT


Managed Factories
-----------------

``#include <raft/core/device_mdspan.hpp>``

.. doxygenfunction:: make_managed_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
    :project: RAFT


Host Vocabulary
---------------

``#include <raft/core/host_mdspan.hpp>``

.. doxygentypedef:: raft::host_mdspan
   :project: RAFT

.. doxygenstruct:: raft::is_host_mdspan
   :project: RAFT

.. doxygentypedef:: raft::is_host_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::is_input_host_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::is_output_host_mdspan_t
   :project: RAFT

.. doxygentypedef:: raft::enable_if_host_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_input_host_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_output_host_mdspan
    :project: RAFT

.. doxygentypedef:: raft::host_matrix_view
   :project: RAFT

.. doxygentypedef:: raft::host_vector_view
   :project: RAFT

.. doxygentypedef:: raft::host_scalar_view
   :project: RAFT

Host Factories
--------------

``#include <raft/core/host_mdspan.hpp>``

.. doxygenfunction:: raft::make_host_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
    :project: RAFT


Validation Routines
-------------------

``#include <raft/core/mdspan.hpp>``

.. doxygenstruct:: raft::is_mdspan
    :project: RAFT
    :members:

.. doxygentypedef:: raft::is_mdspan_t
    :project: RAFT

.. doxygenstruct:: raft::is_input_mdspan
    :project: RAFT
    :members:

.. doxygentypedef:: raft::is_input_mdspan_t
    :project: RAFT

.. doxygenstruct:: raft::is_output_mdspan
    :project: RAFT
    :members:

.. doxygentypedef:: raft::is_output_mdspan_t
    :project: RAFT

.. doxygentypedef:: raft::enable_if_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_input_mdspan
    :project: RAFT

.. doxygentypedef:: raft::enable_if_output_mdspan
    :project: RAFT
