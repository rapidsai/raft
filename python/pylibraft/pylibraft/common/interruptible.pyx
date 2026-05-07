#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import contextlib
import signal

from cuda.bindings.cyruntime cimport cudaStream_t
from cython.operator cimport dereference

from libc.stdint cimport uintptr_t


@contextlib.contextmanager
def cuda_interruptible():
    '''
    Temporarily install a keyboard interrupt handler (Ctrl+C)
    that cancels the enclosed interruptible C++ thread.

    Use this on a long-running C++ function imported via cython:

    >>> with cuda_interruptible():
    >>>     my_long_running_function(...)

    It's also recommended to release the GIL during the call, to
    make sure the handler has a chance to run:

    >>> with cuda_interruptible():
    >>>     with nogil:
    >>>         my_long_running_function(...)
    '''
    cdef shared_ptr[interruptible] token = get_token()

    def newhr(*args, **kwargs):
        with nogil:
            dereference(token).cancel()

    try:
        oldhr = signal.signal(signal.SIGINT, newhr)
    except ValueError:
        # the signal creation would fail if this is not the main thread
        # That's fine! The feature is disabled.
        oldhr = None
    try:
        yield
    finally:
        if oldhr is not None:
            signal.signal(signal.SIGINT, oldhr)


def synchronize(stream):
    '''
    Same as cudaStreamSynchronize, but can be interrupted
    if called within a `with cuda_interruptible()` block.
    Accepts a Stream, int cudaStream_t pointer, or any object
    implementing the __cuda_stream__ protocol.
    '''
    cdef cudaStream_t c_stream_t
    if hasattr(stream, '__cuda_stream__'):
        proto = stream.__cuda_stream__()
        c_stream_t = <cudaStream_t><uintptr_t>proto[1]
    elif isinstance(stream, int):
        c_stream_t = <cudaStream_t><uintptr_t>stream
    else:
        raise TypeError("stream must be a Stream, int, or __cuda_stream__ protocol object")
    with nogil:
        inter_synchronize(c_stream_t)


def cuda_yield():
    '''
    Check for an asynchronously received interrupted_exception.
    Raises the exception if a user pressed Ctrl+C within a
    `with cuda_interruptible()` block before.
    '''
    with nogil:
        inter_yield()
