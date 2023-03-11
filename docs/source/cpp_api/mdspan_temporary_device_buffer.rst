temporary_device_buffer: Temporary raft::device_mdspan Producing Object
===========================================

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/core/temporary_device_buffer.hpp>``

.. doxygenclass:: raft::temporary_device_buffer
    :project: RAFT
    :members:

Factories
---------
.. doxygenfunction:: raft::make_temporary_device_buffer
    :project: RAFT

.. doxygenfunction:: raft::make_readonly_temporary_device_buffer
    :project: RAFT

.. doxygenfunction:: raft::make_writeback_temporary_device_buffer
    :project: RAFT
