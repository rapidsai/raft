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


cdef extern from "raft/comms/comms.hpp" namespace "raft::comms":

    cdef cppclass comms_t:
        pass

    cdef cppclass comms_iface:
        pass




cdef extern from "raft/comms/comms_helper.hpp" namespace "raft::comms":

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


    bool test_collective_allreduce(const handle_t &h) except +
    bool test_pointToPoint_simple_send_recv(const handle_t &h,
                                            int numTrials) except +


def perform_test_comms_allreduce(handle):
    """
    Performs an allreduce on the current worker
    :param handle: Handle handle containing cumlCommunicator to use
    """
    cdef const handle_t* h = <handle_t*><size_t>handle.getHandle()
    return test_collective_allreduce(deref(h))


def perform_test_comms_send_recv(handle, n_trials):
    """
    Performs a p2p send/recv on the current worker
    :param handle: Handle handle containing cumlCommunicator to use
    """
    cdef const handle_t *h = <handle_t*><size_t>handle.getHandle()
    return test_pointToPoint_simple_send_recv(deref(h), <int>n_trials)



def inject_comms_on_handle_coll_only(handle, nccl_inst, size, rank, verbose):
    """
    Given a handle and initialized nccl comm, creates a cumlCommunicator
    instance and injects it into the handle.
    :param handle: Handle cumlHandle to inject comms into
    :param nccl_inst: ncclComm_t initialized nccl comm
    :param size: int number of workers in cluster
    :param rank: int rank of current worker
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
    Given a handle and initialized comms, creates a cumlCommunicator instance
    and injects it into the handle.
    :param handle: Handle cumlHandle to inject comms into
    :param nccl_inst: ncclComm_t initialized nccl comm
    :param ucp_worker: size_t initialized ucp_worker_h instance
    :param eps: size_t array of initialized ucp_ep_h instances
    :param size: int number of workers in cluster
    :param rank: int rank of current worker
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
