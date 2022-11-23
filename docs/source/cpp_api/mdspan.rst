Multi-dimensional Data
======================

This page provides C++ class references for the RAFT's 1d span and multi-dimension owning (mdarray) and non-owning (mdspan) APIs. These headers can be found in the `raft/core` directory.

.. role:: py(code)
   :language: c++
   :class: highlight


Representation
##############


Layouts
-------

.. doxygentypedef:: raft::row_major
    :project: RAFT

.. doxygentypedef:: raft::col_major
    :project: RAFT


Shapes
------

.. doxygentypedef:: raft::matrix_extent
    :project: RAFT

.. doxygentypedef:: raft::vector_extent
    :project: RAFT

.. doxygentypedef:: raft::scalar_extent
    :project: RAFT

.. doxygentypedef:: raft::extent_3d
    :project: RAFT

.. doxygentypedef:: raft::extent_4d
    :project: RAFT

.. doxygentypedef:: raft::extent_5d
    :project: RAFT

.. doxygenfunction:: raft::flatten(mdspan_type mds)
    :project: RAFT

.. doxygenfunction:: raft:: flatten(const array_interface_type& mda)
    :project: RAFT

.. doxygenfunction:: raft::reshape(mdspan_type mds, extents<IndexType, Extents...> new_shape)
    :project: RAFT

.. doxygenfunction:: raft::reshape(const array_interface_type& mda, extents<IndexType, Extents...> new_shape)
    :project: RAFT


Accessors
---------

.. doxygenstruct:: raft::host_device_accessor
    :project: RAFT
    :members:

.. doxygentypedef:: raft::host_accessor
    :project: RAFT

.. doxygentypedef:: raft::device_accessor
    :project: RAFT

.. doxygentypedef:: raft::managed_accessor
    :project: RAFT




mdarray
#######

.. doxygenclass:: raft::mdarray
    :project: RAFT
    :members:

.. doxygenclass:: raft::array_interface
    :project: RAFT
    :members:

.. doxygenstruct:: raft::is_array_interface
    :project: RAFT
    :members:

.. doxygentypedef:: raft::is_array_interface_t
    :project RAFT

Device Vocabulary
-----------------

.. doxygentypedef:: raft::device_mdarray
    :project: RAFT

.. doxygentypedef:: raft::device_matrix
    :project: RAFT

.. doxygentypedef:: raft::device_vector
    :project: RAFT

.. doxygentypedef:: raft::device_scalar
    :project: RAFT


Device Factories
----------------

.. doxygenfunction:: raft::make_device_matrix
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar
    :project: RAFT


Host Vocabulary
---------------

.. doxygentypedef:: raft::host_matrix
    :project: RAFT

.. doxygentypedef:: raft::host_vector
    :project: RAFT

.. doxygentypedef:: raft::host_scalar
    :project: RAFT


Host Factories
--------------

.. doxygenfunction:: raft::make_host_matrix
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar
    :project: RAFT

mdspan
######

.. doxygentypedef:: raft::mdspan
    :project: RAFT

.. doxygenfunction:: raft::make_mdspan
    :project: RAFT

.. doxygenfunction:: raft::make_extents
    :project: RAFT

.. doxygenfunction:: raft::make_strided_layout(Extents extents, Strides strides)
    :project: RAFT

.. doxygenfunction:: raft::unravel_index
    :project: RAFT


Device Vocabulary
-----------------

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

.. doxygenfunction:: raft::make_device_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_vector_view(ElementType* ptr, IndexType n)
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
   :project: RAFT


Managed Vocabulary
------------------

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

.. doxygenfunction:: make_managed_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)


Host Vocabulary
---------------

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

.. doxygenfunction:: raft::make_host_matrix_view
    :project: RAFT

.. doxygenfunction:: raft::make_host_vector_view
    :project: RAFT

.. doxygenfunction:: raft::make_device_scalar_view
    :project: RAFT


Validation Routines
-------------------

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

span
####

.. doxygentypedef:: raft::device_span
   :project: RAFT

.. doxygentypedef:: raft::host_span
   :project: RAFT

.. doxygenclass:: raft::span
    :project: RAFT
    :members:
