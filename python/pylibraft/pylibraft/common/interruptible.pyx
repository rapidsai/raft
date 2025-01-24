#
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import contextlib
import signal

from cuda.bindings.cyruntime cimport cudaStream_t
from cython.operator cimport dereference

from rmm.librmm.cuda_stream_view cimport cuda_stream_view

from .cuda cimport Stream


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


def synchronize(stream: Stream):
    '''
    Same as cudaStreamSynchronize, but can be interrupted
    if called within a `with cuda_interruptible()` block.
    '''
    cdef cuda_stream_view c_stream = cuda_stream_view(stream.getStream())
    with nogil:
        inter_synchronize(c_stream)


def cuda_yield():
    '''
    Check for an asynchronously received interrupted_exception.
    Raises the exception if a user pressed Ctrl+C within a
    `with cuda_interruptible()` block before.
    '''
    with nogil:
        inter_yield()
