#
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from cython.operator cimport dereference as deref


@contextlib.contextmanager
def cuda_interruptible():
    '''
    Temporarily install a keyboard interrupt handler (Ctrl+C)
    that cancels the enclosed interruptible C++ thread.

    Use this on a long-running C++ function imported via cython:

    .. code-block:: python

        with cuda_interruptible():
            my_long_running_function(...)

    It's also recommended to release the GIL during the call, to
    make sure the handler has a chance to run:

    .. code-block:: python

        with cuda_interruptible():
            with nogil:
                my_long_running_function(...)

    '''
    cdef shared_ptr[interruptible] token = get_token()

    def newhr(*args, **kwargs):
        with nogil:
            deref(token).cancel_no_throw()

    oldhr = signal.signal(signal.SIGINT, newhr)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, oldhr)
