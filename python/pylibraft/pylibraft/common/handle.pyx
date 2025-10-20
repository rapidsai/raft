#
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import functools

from cuda.bindings.cyruntime cimport cudaStream_t
from libc.stdint cimport uintptr_t

from rmm.librmm.cuda_stream_view cimport (
    cuda_stream_per_thread,
    cuda_stream_view,
)

from .cuda cimport Stream

from .cuda import CudaRuntimeError


cdef class DeviceResources:
    """
    DeviceResources is a lightweight python wrapper around the corresponding
    C++ class of device_resources exposed by RAFT's C++ interface. Refer to
    the header file raft/core/device_resources.hpp for interface level
    details of this struct

    Parameters
    ----------
    stream : Optional stream to use for ordering CUDA instructions
             Accepts pylibraft.common.Stream() or uintptr_t (cudaStream_t)

    Examples
    --------

    Basic usage:

    >>> from pylibraft.common import Stream, DeviceResources
    >>> stream = Stream()
    >>> handle = DeviceResources(stream)
    >>>
    >>> # call algos here
    >>>
    >>> # final sync of all work launched in the stream of this handle
    >>> # this is same as `raft.cuda.Stream.sync()` call, but safer in case
    >>> # the default stream inside the `device_resources` is being used
    >>> handle.sync()
    >>> del handle  # optional!

    Using a cuPy stream with RAFT device_resources:

    >>> import cupy
    >>> from pylibraft.common import Stream, DeviceResources
    >>>
    >>> cupy_stream = cupy.cuda.Stream()
    >>> handle = DeviceResources(stream=cupy_stream.ptr)

    Using a RAFT stream with CuPy ExternalStream:

    >>> import cupy
    >>> from pylibraft.common import Stream
    >>>
    >>> raft_stream = Stream()
    >>> cupy_stream = cupy.cuda.ExternalStream(raft_stream.get_ptr())
    """

    def __cinit__(self, stream=None, n_streams=0):
        self.n_streams = n_streams

        if n_streams > 0:
            self.stream_pool.reset(new cuda_stream_pool(n_streams))

        cdef uintptr_t s
        cdef cuda_stream_view c_stream

        # We should either have a pylibraft.common.Stream or a uintptr_t
        # of a cudaStream_t
        if stream is None:
            # this constructor will construct a "main" handle on
            # per-thread default stream, which is non-blocking
            self.c_obj.reset(new handle_t(cuda_stream_per_thread,
                                          self.stream_pool))
        else:
            # this constructor constructs a handle on user stream
            if isinstance(stream, Stream):
                # Stream is pylibraft Stream()
                s = stream.get_ptr()
                c_stream = cuda_stream_view(<cudaStream_t>s)
            elif isinstance(stream, int):
                # Stream is a pointer, cast to cudaStream_t
                s = stream
                c_stream = cuda_stream_view(<cudaStream_t>s)
            else:
                raise ValueError("stream should be common.Stream() or "
                                 "uintptr_t to cudaStream_t")

            self.c_obj.reset(new handle_t(c_stream,
                             self.stream_pool))

    def sync(self):
        """
        Issues a sync on the stream set for this instance.
        """
        self.c_obj.get()[0].sync_stream()

    def getHandle(self):
        """
        Return the pointer to the underlying raft::device_resources
        instance as a size_t
        """
        return <size_t> self.c_obj.get()

    def __getstate__(self):
        return self.n_streams

    def __setstate__(self, state):
        self.n_streams = state
        if self.n_streams > 0:
            self.stream_pool.reset(new cuda_stream_pool(self.n_streams))

        self.c_obj.reset(new device_resources(cuda_stream_per_thread,
                                              self.stream_pool))


cdef class Handle(DeviceResources):
    """
    Handle is a lightweight python wrapper around the corresponding
    C++ class of handle_t exposed by RAFT's C++ interface. Refer to
    the header file raft/core/handle.hpp for interface level
    details of this struct

    Note: This API is officially deprecated in favor of DeviceResources
    and will be removed in a future release.

    Parameters
    ----------
    stream : Optional stream to use for ordering CUDA instructions
            Accepts pylibraft.common.Stream() or uintptr_t (cudaStream_t)

    Examples
    --------

    Basic usage:

    >>> from pylibraft.common import Stream, Handle
    >>> stream = Stream()
    >>> handle = Handle(stream)
    >>>
    >>> # call algos here
    >>>
    >>> # final sync of all work launched in the stream of this handle
    >>> # this is same as `raft.cuda.Stream.sync()` call, but safer in case
    >>> # the default stream inside the `handle_t` is being used
    >>> handle.sync()
    >>> del handle  # optional!

    Using a cuPy stream with RAFT device_resources:

    >>> import cupy
    >>> from pylibraft.common import Stream, Handle
    >>>
    >>> cupy_stream = cupy.cuda.Stream()
    >>> handle = Handle(stream=cupy_stream.ptr)

    Using a RAFT stream with CuPy ExternalStream:

    >>> import cupy
    >>> from pylibraft.common import Stream
    >>>
    >>> raft_stream = Stream()
    >>> cupy_stream = cupy.cuda.ExternalStream(raft_stream.get_ptr())

    """
    def __getstate__(self):
        return self.n_streams

    def __setstate__(self, state):
        self.n_streams = state
        if self.n_streams > 0:
            self.stream_pool.reset(new cuda_stream_pool(self.n_streams))

        self.c_obj.reset(new handle_t(cuda_stream_per_thread,
                                      self.stream_pool))


_HANDLE_PARAM_DOCSTRING = """
     handle : Optional RAFT resource handle for reusing CUDA resources.
        If a handle isn't supplied, CUDA resources will be
        allocated inside this function and synchronized before the
        function exits. If a handle is supplied, you will need to
        explicitly synchronize yourself by calling `handle.sync()`
        before accessing the output.
""".strip()


def auto_sync_handle(f):
    """Decorator to automatically call sync on a raft handle when
    it isn't passed to a function.

    When a handle=None is passed to the wrapped function, this decorator
    will automatically create a default handle for the function, and
    call sync on that handle when the function exits.

    This will also insert the appropriate docstring for the handle parameter
    """

    @functools.wraps(f)
    def wrapper(*args, handle=None, **kwargs):
        sync_handle = handle is None
        handle = handle if handle is not None else DeviceResources()

        ret_value = f(*args, handle=handle, **kwargs)

        if sync_handle:
            handle.sync()

        return ret_value

    wrapper.__doc__ = wrapper.__doc__.format(
        handle_docstring=_HANDLE_PARAM_DOCSTRING
    )
    return wrapper


cdef class DeviceResourcesSNMG:
    """
    DeviceResourcesSNMG manages multi-GPU resources
    in a single-node setup using RAFT's device_resources_snmg. Refer to
    the header file raft/core/device_resources_snmg.hpp for interface level
    details of this struct
    Parameters
    ----------
    device_ids : Optional list to specify which devices will be used
    Examples
    --------
    Basic usage:
    >>> from pylibraft.common import DeviceResourcesSNMG
    >>>
    >>> # to use GPU IDs 0,1,2,3 on machine
    >>> # handle = DeviceResourcesSNMG([0,1,2,3])
    >>>
    >>> # to use all GPUs on machine
    >>> handle = DeviceResourcesSNMG()
    """

    def __cinit__(self, device_ids=None):
        self.device_ids = device_ids

        cdef device_resources_snmg* snmg_ptr
        cdef vector[int] ids

        if self.device_ids is None:
            snmg_ptr = new device_resources_snmg()
        else:
            for id in self.device_ids:
                ids.push_back(id)
            snmg_ptr = new device_resources_snmg(ids)

        self.c_obj.reset(snmg_ptr)

    def sync(self):
        """
        Issues a sync on the stream set for this instance.
        """
        self.c_obj.get()[0].sync_stream()

    def getHandle(self):
        """
        Return the pointer to the underlying raft::device_resources_snmg
        instance as a size_t
        """
        return <size_t> self.c_obj.get()

    def __getstate__(self):
        return self.device_ids

    def __setstate__(self, state):
        self.device_ids = state

        cdef device_resources_snmg* snmg_ptr
        cdef vector[int] ids

        if self.device_ids is None:
            snmg_ptr = new device_resources_snmg()
        else:
            for id in self.device_ids:
                ids.push_back(id)
            snmg_ptr = new device_resources_snmg(ids)

        self.c_obj.reset(snmg_ptr)
