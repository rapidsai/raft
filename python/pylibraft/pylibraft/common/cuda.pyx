#
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuda.bindings.cyruntime cimport (
    cudaError_t,
    cudaGetErrorName,
    cudaGetErrorString,
    cudaGetLastError,
    cudaStream_t,
    cudaStreamCreate,
    cudaStreamDestroy,
    cudaStreamSynchronize,
    cudaSuccess,
)
from libc.stdint cimport uintptr_t


class CudaRuntimeError(RuntimeError):
    def __init__(self, extraMsg=None):
        cdef cudaError_t e = cudaGetLastError()
        cdef bytes errMsg = cudaGetErrorString(e)
        cdef bytes errName = cudaGetErrorName(e)
        msg = "Error! %s reason='%s'" % (errName.decode(), errMsg.decode())
        if extraMsg is not None:
            msg += " extraMsg='%s'" % extraMsg
        super(CudaRuntimeError, self).__init__(msg)


cdef class Stream:
    """
    Stream represents a thin-wrapper around cudaStream_t and its operations.

    Examples
    --------

    >>> from pylibraft.common.cuda import Stream
    >>> stream = Stream()
    >>> stream.sync()
    >>> del stream  # optional!
    """
    def __cinit__(self):
        cdef cudaStream_t stream
        cdef cudaError_t e = cudaStreamCreate(&stream)
        if e != cudaSuccess:
            raise CudaRuntimeError("Stream create")
        self.s = stream

    def __dealloc__(self):
        self.sync()
        cdef cudaError_t e = cudaStreamDestroy(self.s)
        if e != cudaSuccess:
            raise CudaRuntimeError("Stream destroy")

    def sync(self):
        """
        Synchronize on the cudastream owned by this object. Note that this
        could raise exception due to issues with previous asynchronous
        launches
        """
        cdef cudaError_t e = cudaStreamSynchronize(self.s)
        if e != cudaSuccess:
            raise CudaRuntimeError("Stream sync")

    cdef cudaStream_t getStream(self):
        return self.s

    def get_ptr(self):
        """
        Return the uintptr_t pointer of the underlying cudaStream_t handle
        """
        return <uintptr_t>self.s
