Common
======

This page provides `pylibraft` class references for the publicly-exposed elements of the `pylibraft.common` package.


.. role:: py(code)
   :language: python
   :class: highlight


Basic Vocabulary
################

.. autoclass:: pylibraft.common.DeviceResources
    :members:

.. autoclass:: pylibraft.common.Stream
    :members:

.. autoclass:: pylibraft.common.device_ndarray
    :members:

Interruptible
#############

.. autofunction:: pylibraft.common.interruptible.cuda_interruptible

.. autofunction:: pylibraft.common.interruptible.synchronize

.. autofunction:: pylibraft.common.interruptible.cuda_yield


CUDA Array Interface Helpers
############################

.. autoclass:: pylibraft.common.cai_wrapper
    :members:
