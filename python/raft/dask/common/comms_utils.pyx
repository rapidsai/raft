# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref

from cpython.long cimport PyLong_AsVoidPtr

from libcpp cimport bool

from libc.stdint cimport uintptr_t

cdef extern from "nccl.h":

    cdef struct ncclComm
    ctypedef ncclComm *ncclComm_t

cdef extern from "raft/handle.hpp" namespace "raft":
    cdef cppclass handle_t:
        handle_t() except +

cdef extern from "raft/comms/std_comms.hpp" namespace "raft::comms":

    cdef cppclass std_comms:
        pass

cdef extern from "raft/comms/helper.hpp" namespace "raft::comms":

    void build_comms_nccl_ucx(handle_t *handle,
                              ncclComm_t comm,
                              void *ucp_worker,
                              void *eps,
                              int size,
                              int rank) except +

    void build_comms_nccl_only(handle_t *handle,
                               ncclComm_t comm,
                               int size,
                               int rank) except +

cdef extern from "raft/comms/test.hpp" namespace "raft::comms":

    bool test_collective_allreduce(const handle_t &h, int root) except +
    bool test_collective_broadcast(const handle_t &h, int root) except +
    bool test_collective_reduce(const handle_t &h, int root) except +
    bool test_collective_allgather(const handle_t &h, int root) except +
    bool test_collective_gather(const handle_t &h, int root) except +
    bool test_collective_gatherv(const handle_t &h, int root) except +
    bool test_collective_reducescatter(const handle_t &h, int root) except +
    bool test_pointToPoint_simple_send_recv(const handle_t &h,
                                            int numTrials) except +
    bool test_pointToPoint_device_send_or_recv(const handle_t &h,
                                               int numTrials) except +
    bool test_pointToPoint_device_sendrecv(const handle_t &h,
                                           int numTrials) except +
    bool test_pointToPoint_device_multicast_sendrecv(const handle_t &h,
                                                     int numTrials) except +
    bool test_commsplit(const handle_t &h, int n_colors) except +


def perform_test_comms_allreduce(handle, root):
    """
    Performs an allreduce on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_allreduce(deref(h), root)


def perform_test_comms_reduce(handle, root):
    """
    Performs an allreduce on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_reduce(deref(h), root)


def perform_test_comms_reducescatter(handle, root):
    """
    Performs an allreduce on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_reducescatter(deref(h), root)


def perform_test_comms_bcast(handle, root):
    """
    Performs an broadcast on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_broadcast(deref(h), root)


def perform_test_comms_allgather(handle, root):
    """
    Performs an broadcast on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_allgather(deref(h), root)


def perform_test_comms_gather(handle, root):
    """
    Performs a gather on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    root : int
           Rank of the root worker
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_gather(deref(h), root)


def perform_test_comms_gatherv(handle, root):
    """
    Performs a gatherv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    root : int
           Rank of the root worker
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_gatherv(deref(h), root)


def perform_test_comms_send_recv(handle, n_trials):
    """
    Performs a p2p send/recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const handle_t *h = <handle_t*><size_t>handle.getHandle()
    return test_pointToPoint_simple_send_recv(deref(h), <int>n_trials)


def perform_test_comms_device_send_or_recv(handle, n_trials):
    """
    Performs a p2p device send or recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const handle_t *h = <handle_t*><size_t>handle.getHandle()
    return test_pointToPoint_device_send_or_recv(deref(h), <int>n_trials)


def perform_test_comms_device_sendrecv(handle, n_trials):
    """
    Performs a p2p device concurrent send&recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const handle_t *h = <handle_t*><size_t>handle.getHandle()
    return test_pointToPoint_device_sendrecv(deref(h), <int>n_trials)


def perform_test_comms_device_multicast_sendrecv(handle, n_trials):
    """
    Performs a p2p device concurrent multicast send&recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const handle_t *h = <handle_t *> <size_t> handle.getHandle()
    return test_pointToPoint_device_multicast_sendrecv(deref(h), <int>n_trials)


def perform_test_comm_split(handle, n_colors):
    """
    Performs a p2p send/recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const handle_t * h = < handle_t * > < size_t > handle.getHandle()
    return test_commsplit(deref(h), < int > n_colors)


def inject_comms_on_handle_coll_only(handle, nccl_inst, size, rank, verbose):
    """
    Given a handle and initialized nccl comm, creates a comms_t
    instance and injects it into the handle.

        Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    nccl_inst : raft.dask.common.nccl
                Initialized nccl comm to use
    size : int
           Number of workers in cluster
    rank : int
           Rank of current worker

    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t

    cdef size_t nccl_comm_size_t = <size_t>nccl_inst.get_comm()
    nccl_comm_ = <ncclComm_t*>nccl_comm_size_t

    build_comms_nccl_only(handle_,
                          deref(nccl_comm_),
                          size,
                          rank)


def inject_comms_on_handle(handle, nccl_inst, ucp_worker, eps, size,
                           rank, verbose):
    """
    Given a handle and initialized comms, creates a comms_t instance
    and injects it into the handle.

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    nccl_inst : raft.dask.common.nccl
                Initialized nccl comm to use
    ucp_worker : size_t pointer to initialized ucp_worker_h instance
    eps: size_t pointer to array of initialized ucp_ep_h instances
    size : int
           Number of workers in cluster
    rank : int
           Rank of current worker
    """
    cdef size_t *ucp_eps = <size_t*> malloc(len(eps)*sizeof(size_t))

    for i in range(len(eps)):
        if eps[i] is not None:
            ep_st = <uintptr_t>eps[i].get_ucp_endpoint()
            ucp_eps[i] = <size_t>ep_st
        else:
            ucp_eps[i] = 0

    cdef void* ucp_worker_st = <void*><size_t>ucp_worker

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t

    cdef size_t nccl_comm_size_t = <size_t>nccl_inst.get_comm()
    nccl_comm_ = <ncclComm_t*>nccl_comm_size_t

    build_comms_nccl_ucx(handle_,
                         deref(nccl_comm_),
                         <void*>ucp_worker_st,
                         <void*>ucp_eps,
                         size,
                         rank)

    free(ucp_eps)
