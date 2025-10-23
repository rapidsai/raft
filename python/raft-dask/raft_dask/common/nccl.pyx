#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from libcpp cimport bool


cdef extern from "raft/comms/std_comms.hpp" namespace "raft::comms":
    void get_nccl_unique_id(char *uid) except +
    void nccl_unique_id_from_char(ncclUniqueId *id,
                                  char *uniqueId) except +

cdef extern from "nccl.h":

    cdef struct ncclComm

    ctypedef struct ncclUniqueId:
        char *internal[128]

    ctypedef ncclComm *ncclComm_t

    ctypedef enum ncclResult_t:
        ncclSuccess
        ncclUnhandledCudaError
        ncclSystemError
        ncclInternalError
        ncclInvalidArgument
        ncclInvalidUsage
        ncclInProgress
        ncclNumResults

    ncclResult_t ncclCommInitRank(ncclComm_t *comm,
                                  int nranks,
                                  ncclUniqueId commId,
                                  int rank) nogil

    ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) nogil

    ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) nogil

    ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *count) nogil

    const char *ncclGetErrorString(ncclResult_t result) nogil

    ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) nogil

    ncclResult_t ncclCommAbort(ncclComm_t comm) nogil

    ncclResult_t ncclCommFinalize(ncclComm_t comm) nogil

    ncclResult_t ncclCommDestroy(ncclComm_t comm) nogil

NCCL_UNIQUE_ID_BYTES = 128


def unique_id():
    """
    Returns a new ncclUniqueId converted to a
    character array that can be safely serialized
    and shared to a remote worker.

    Returns
    -------
    128-byte unique id : str
    """
    cdef char *uid = <char *> malloc(NCCL_UNIQUE_ID_BYTES * sizeof(char))
    get_nccl_unique_id(uid)
    c_str = uid[:NCCL_UNIQUE_ID_BYTES-1]
    c_str
    free(uid)
    return c_str


cdef class nccl:
    """
    A NCCL wrapper for initializing and closing NCCL comms
    in Python.
    """
    cdef ncclComm_t *comm

    cdef int size
    cdef int rank

    def __cinit__(self):
        self.comm = <ncclComm_t*>malloc(sizeof(ncclComm_t))

    def __dealloc__(self):

        comm_ = <ncclComm_t*>self.comm

        if comm_ != NULL:
            free(self.comm)
            self.comm = NULL

    @staticmethod
    def get_unique_id():
        """
        Returns a new nccl unique id

        Returns
        -------
        nccl unique id : str
        """
        return unique_id()

    def init(self, nranks, commId, rank):
        """
        Construct a nccl-py object

        Parameters
        ----------
        nranks : int size of clique
        commId : string unique id from client
        rank : int rank of current worker
        """
        self.size = nranks
        self.rank = rank

        cdef ncclUniqueId *ident = <ncclUniqueId*>malloc(sizeof(ncclUniqueId))
        nccl_unique_id_from_char(ident, commId)

        comm_ = <ncclComm_t*>self.comm

        cdef int nr = nranks
        cdef int r = rank
        cdef ncclResult_t result

        with nogil:
            result = ncclCommInitRank(comm_, nr,
                                      deref(ident), r)

        if result != ncclSuccess:
            with nogil:
                err_str = ncclGetErrorString(result)

            raise RuntimeError("NCCL_ERROR: %s" % err_str)

    def destroy(self):
        """
        Call finalize and destroy on the underlying NCCL comm.
        
        ncclCommFinalize is an intra-node collective operation that must be
        called before ncclCommDestroy to properly finalize internal NCCL state
        and avoid hangs, especially in multi-node setups.
        
        Since ncclCommFinalize is asynchronous, we poll until completion before
        calling ncclCommDestroy.
        
        See: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html
        """
        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result
        cdef ncclResult_t asyncErr
        if comm_ != NULL:
            # Step 1: Finalize (intra-node collective, asynchronous)
            with nogil:
                result = ncclCommFinalize(deref(comm_))

            if result != ncclSuccess and result != ncclInProgress:
                with nogil:
                    err_str = ncclGetErrorString(result)
                raise RuntimeError("NCCL_ERROR during finalize: %s" % err_str)

            # Step 2: Poll until finalize completes
            # ncclCommFinalize returns immediately with ncclInProgress,
            # so we need to poll ncclCommGetAsyncError until it's done
            with nogil:
                asyncErr = ncclInProgress
                while asyncErr == ncclInProgress:
                    result = ncclCommGetAsyncError(deref(comm_), &asyncErr)
                    if result != ncclSuccess:
                        break

            # Check for errors during finalize
            if asyncErr != ncclSuccess:
                with nogil:
                    err_str = ncclGetErrorString(asyncErr)
                raise RuntimeError("NCCL_ERROR during finalize (async): %s" % err_str)

            # Step 3: Destroy (now safe after finalize completed)
            with nogil:
                result = ncclCommDestroy(deref(comm_))

            free(self.comm)
            self.comm = NULL

            if result != ncclSuccess:
                with nogil:
                    err_str = ncclGetErrorString(result)
                raise RuntimeError("NCCL_ERROR during destroy: %s" % err_str)

    def abort(self):
        """
        Call abort on the underlying nccl comm
        """
        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result
        if comm_ != NULL:
            with nogil:
                result = ncclCommAbort(deref(comm_))

            free(comm_)
            self.comm = NULL

            if result != ncclSuccess:
                with nogil:
                    err_str = ncclGetErrorString(result)
                raise RuntimeError("NCCL_ERROR: %s" % err_str)

    def cu_device(self):
        """
        Get the device backing the underlying comm

        Returns
        -------
        device id : int
        """
        cdef int *dev = <int*>malloc(sizeof(int))

        comm_ = <ncclComm_t*>self.comm
        cdef ncclResult_t result
        with nogil:
            result = ncclCommCuDevice(deref(comm_), dev)

        ret = dev[0]
        free(dev)

        if result != ncclSuccess:
            with nogil:
                err_str = ncclGetErrorString(result)

            raise RuntimeError("NCCL_ERROR: %s" % err_str)

        return ret

    def user_rank(self):
        """
        Get the rank id of the current comm

        Returns
        -------
        rank : int
        """

        cdef int *rank = <int*>malloc(sizeof(int))

        comm_ = <ncclComm_t*>self.comm

        cdef ncclResult_t result
        with nogil:
            result = ncclCommUserRank(deref(comm_), rank)

        ret = rank[0]
        free(rank)

        if result != ncclSuccess:
            with nogil:
                err_str = ncclGetErrorString(result)
            raise RuntimeError("NCCL_ERROR: %s" % err_str)

        return ret

    def get_comm(self):
        """
        Returns the underlying nccl comm in a size_t (similar to void*).
        This can be safely typecasted from size_t into ncclComm_t*

        Returns
        -------
        ncclComm_t instance pointer : size_t
        """
        return <size_t>self.comm
