# Copyright (c) 2020, NVIDIA CORPORATION.
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

from .nccl import nccl
from .ucx import UCX

from .comms_utils import inject_comms_on_handle
from .comms_utils import inject_comms_on_handle_coll_only

from .utils import parse_host_port
from ...common.handle import Handle

from dask.distributed import get_worker, default_client

import warnings

import logging
import time
import uuid
from collections import OrderedDict

logger = logging.getLogger(__name__)


class Comms:

    """
    Initializes and manages underlying NCCL and UCX comms handles across
    the workers of a Dask cluster. It is expected that `init()` will be
    called explicitly. It is recommended to also call `destroy()` when
    the comms are no longer needed so the underlying resources can be
    cleaned up. This class is not meant to be thread-safe.

    Examples
    --------
   .. code-block:: python

        # The following code block assumes we have wrapped a C++
        # function in a Python function called `run_algorithm`,
        # which takes a `raft::handle_t` as a single argument.
        # Once the `Comms` instance is successfully initialized,
        # the underlying `raft::handle_t` will contain an instance
        # of `raft::comms::comms_t`

        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client

        from raft.dask.common import Comms, local_handle

        cluster = LocalCUDACluster()
        client = Client(cluster)

        def _use_comms(sessionId):
            return run_algorithm(local_handle(sessionId))

        comms = Comms(client=client)
        comms.init()

        futures = [client.submit(_use_comms,
                                 comms.sessionId,
                                 workers=[w],
                                 pure=False) # Don't memoize
                       for w in cb.worker_addresses]
        wait(dfs, timeout=5)

        comms.destroy()
        client.close()
        cluster.close()
    """

    valid_nccl_placements = ('client', 'worker', 'scheduler')

    def __init__(self, comms_p2p=False, client=None, verbose=False,
                 streams_per_handle=0, nccl_root_location="scheduler"):
        """
        Construct a new CommsContext instance

        Parameters
        ----------
        comms_p2p : bool
                    Initialize UCX endpoints?
        client : dask.distributed.Client [optional]
                 Dask client to use
        verbose : bool
                  Print verbose logging
        nccl_root_location : string
                  Indicates where the NCCL's root node should be located. ['client', 'worker', 'scheduler' (default)]

        """
        self.client = client if client is not None else default_client()

        self.comms_p2p = comms_p2p

        if (nccl_root_location.lower() not in Comms.valid_nccl_placements):
            raise ValueError(f"nccl_root_location must be one of: {Comms.valid_nccl_placements}")
        self.nccl_root_location = nccl_root_location.lower()

        self.streams_per_handle = streams_per_handle

        self.sessionId = uuid.uuid4().bytes

        self.nccl_initialized = False
        self.ucx_initialized = False

        self.verbose = verbose

        if verbose:
            print("Initializing comms!")

    def __del__(self):
        if self.nccl_initialized or self.ucx_initialized:
            self.destroy()

    def worker_info(self, workers):
        """
        Builds a dictionary of { (worker_address, worker_port) :
                                (worker_rank, worker_port ) }
        """
        ranks = _func_worker_ranks(workers)
        ports = _func_ucp_ports(self.client, workers) \
            if self.comms_p2p else None

        output = {}
        for k in ranks.keys():
            output[k] = {"rank": ranks[k]}
            if self.comms_p2p:
                output[k]["port"] = ports[k]
        return output

    def init(self, workers=None):
        """
        Initializes the underlying comms. NCCL is required but
        UCX is only initialized if `comms_p2p == True`

        Parameters
        ----------

        workers : Sequence
                  Unique collection of workers for initializing comms.
        """

        self.worker_addresses = list(OrderedDict.fromkeys(
            self.client.scheduler_info()["workers"].keys()
            if workers is None else workers))

        if self.nccl_initialized or self.ucx_initialized:
            warnings.warn("Comms have already been initialized.")
            return

        worker_info = self.worker_info(self.worker_addresses)
        worker_info = {w: worker_info[w] for w in self.worker_addresses}

        if (self.nccl_root_location == 'client'):
            self.uniqueId = nccl.get_unique_id()
        elif (self.nccl_root_location == 'worker'):
            self.uniqueId = self.client.run(_func_set_worker_as_nccl_root,
                                            sessionId=self.sessionId,
                                            verbose=self.verbose,
                                            workers=[self.worker_addresses[0]],
                                            wait=True)[self.worker_addresses[0]]
        else:
            self.uniqueId = self.client.run_on_scheduler(_func_set_scheduler_as_nccl_root,
                                                         sessionId=self.sessionId,
                                                         verbose=self.verbose)

        self.client.run(_func_init_all,
                        self.sessionId,
                        self.uniqueId,
                        self.comms_p2p,
                        worker_info,
                        self.verbose,
                        self.streams_per_handle,
                        workers=self.worker_addresses,
                        wait=True)

        self.nccl_initialized = True

        if self.comms_p2p:
            self.ucx_initialized = True

        if self.verbose:
            print("Initialization complete.")

    def destroy(self):
        """
        Shuts down initialized comms and cleans up resources. This will
        be called automatically by the Comms destructor, but may be called
        earlier to save resources.
        """
        self.client.run(_func_destroy_all,
                        self.sessionId,
                        self.comms_p2p,
                        self.verbose,
                        wait=True,
                        workers=self.worker_addresses)

        if (self.nccl_root_location == 'scheduler'):
            self.client.run_on_scheduler(_func_destroy_scheduler_session,
                                         self.sessionId)

        if self.verbose:
            print("Destroying comms.")

        self.nccl_initialized = False
        self.ucx_initialized = False


def local_handle(sessionId):
    """
    Simple helper function for retrieving the local handle_t instance
    for a comms session on a worker.

    Parameters
    ----------
    sessionId : str
                session identifier from an initialized comms instance

    Returns
    -------

    handle : raft.Handle or None
    """
    state = worker_state(sessionId)
    return state["handle"] if "handle" in state else None


def scheduler_state(sessionId, dask_scheduler):
    """
    Retrieves cuML comms state on the scheduler node, for the given sessionId, creating
    a new session if it does not exist. If no session id is given, returns the state dict for
    all sessions.

    Parameters
    ----------
    sessionId : SessionId value to retrieve from the dask_scheduler instances
    dask_scheduler : Dask Scheduler object

    Returns
    -------

    session state : str
                    session state associated with sessionId
    """

    if (not hasattr(dask_scheduler, "_raft_comm_state")):
        dask_scheduler._raft_comm_state = {}

    if (sessionId is not None and sessionId not in dask_scheduler._raft_comm_state):
        dask_scheduler._raft_comm_state[sessionId] = {"ts": time.time()}

        return dask_scheduler._raft_comm_state[sessionId]

    return dask_scheduler._raft_comm_state


def worker_state(sessionId=None):
    """
    Retrieves cuML comms state on local worker for the given
    sessionId, creating a new session if it does not exist.
    If no session id is given, returns the state dict for all
    sessions.

    Parameters
    ----------
    sessionId : str
                session identifier from initialized comms instance
    """
    worker = get_worker()
    if not hasattr(worker, "_raft_comm_state"):
        worker._raft_comm_state = {}
    if sessionId is not None and sessionId not in worker._raft_comm_state:
        # Build state for new session and mark session creation time
        worker._raft_comm_state[sessionId] = {"ts": time.time()}

    if sessionId is not None:
        return worker._raft_comm_state[sessionId]
    return worker._raft_comm_state


def get_ucx():
    """
    A simple convenience wrapper to make sure UCP listener and
    endpoints are only ever assigned once per worker.
    """
    if "ucx" not in worker_state("ucp"):
        worker_state("ucp")["ucx"] = UCX.get()
    return worker_state("ucp")["ucx"]

def _func_destroy_scheduler_session(sessionId, dask_scheduler):
    """
    Remove session date from _raft_comm_state, associated with sessionId

    Parameters
    ----------
    sessionId : session Id to be destroyed.
    dask_scheduler : dask_scheduler object (Note: this is supplied by DASK, not the client)
    """
    if (sessionId is not None and sessionId in dask_scheduler._raft_comm_state):
        del dask_scheduler._raft_comm_state[sessionId]
    else:
        return 1

    return 0

def _func_set_scheduler_as_nccl_root(sessionId, verbose, dask_scheduler):
    """
    Creates a persistent nccl uniqueId on the scheduler node.


    Parameters
    ----------
    sessionId : Associated session to attach the unique ID to.
    verbose : Indicates whether or not to emit additional information
    dask_scheduler : dask scheduler object, (Note: this is supplied by DASK, not the client)

    Return
    ------
    uniqueId : byte str
                NCCL uniqueId, associating the DASK scheduler as its root node.
    """
    if(verbose):
        logger.info(msg=f"Setting scheduler as NCCL root for sessionId, '{sessionId}'")

    if (sessionId is None):
        raise ValueError("sessionId cannot be None.")

    session_state = scheduler_state(sessionId=sessionId, dask_scheduler=dask_scheduler)
    if ('nccl_uid' not in session_state):
        session_state['nccl_uid'] = nccl.get_unique_id()

    if(verbose):
        logger.info(f"Done setting scheduler as NCCL root.")

    return session_state['nccl_uid']

def _func_set_worker_as_nccl_root(sessionId, verbose):
    """
    Creates a persistent nccl uniqueId on the scheduler node.


    Parameters
    ----------
    sessionId : Associated session to attach the unique ID to.
    verbose : Indicates whether or not to emit additional information

    Return
    ------
    uniqueId : byte str
                NCCL uniqueId, associating this DASK worker as its root node.
    """
    if(verbose):
        get_worker().log_event(topic="info",
                               msg=f"Setting worker as NCCL root for session, '{sessionId}'")

    if (sessionId is None):
        raise ValueError("sessionId cannot be None.")

    session_state = worker_state(sessionId)
    if ('nccl_uid' not in session_state):
        session_state['nccl_uid'] = nccl.get_unique_id()

    if(verbose):
        get_worker().log_event(topic="info",
                               msg=f"Done setting scheduler as NCCL root.")

    return session_state['nccl_uid']

def _func_ucp_listener_port():
    return get_ucx().listener_port()


async def _func_init_all(sessionId, uniqueId, comms_p2p,
                         worker_info, verbose, streams_per_handle):

    session_state = worker_state(sessionId)
    session_state["nccl_uid"] = uniqueId
    session_state["wid"] = worker_info[get_worker().address]["rank"]
    session_state["nworkers"] = len(worker_info)

    if verbose:
        get_worker().log_event(topic="info", msg="Initializing NCCL.")
        start = time.time()

    _func_init_nccl(sessionId, uniqueId)

    if verbose:
        elapsed = time.time() - start
        get_worker().log_event(topic="info", msg=f"NCCL Initialization took: {elapsed} seconds.")

    if comms_p2p:
        if verbose:
            get_worker().log_event(topic="info", msg="Initializing UCX Endpoints")

        if verbose:
            start = time.time()
        await _func_ucp_create_endpoints(sessionId, worker_info)

        if verbose:
            elapsed = time.time() - start
            msg = f"Done initializing UCX endpoints. Took: {elapsed} seconds.\nBuilding handle."
            get_worker().log_event(topic="info", msg=msg)

        _func_build_handle_p2p(sessionId, streams_per_handle, verbose)

        if verbose:
            get_worker().log_event(topic="info", msg="Done building handle.")

    else:
        _func_build_handle(sessionId, streams_per_handle, verbose)


def _func_init_nccl(sessionId, uniqueId):
    """
    Initialize ncclComm_t on worker

    Parameters
    ----------
    sessionId : str
                session identifier from a comms instance
    uniqueId : array[byte]
               The NCCL unique Id generated from the
               client.
    """

    wid = worker_state(sessionId)["wid"]
    nWorkers = worker_state(sessionId)["nworkers"]

    try:
        n = nccl()
        n.init(nWorkers, uniqueId, wid)
        worker_state(sessionId)["nccl"] = n
    except Exception as e:
        get_worker().log_event(topic="error", msg="An error occurred initializing NCCL!.")
        raise


def _func_build_handle_p2p(sessionId, streams_per_handle, verbose):
    """
    Builds a handle_t on the current worker given the initialized comms

    Parameters
    ----------
    sessionId : str id to reference state for current comms instance.
    streams_per_handle : int number of internal streams to create
    verbose : bool print verbose logging output
    """
    if (verbose):
        get_worker().log_event(topic="info", msg="Building p2p handle.")

    ucp_worker = get_ucx().get_worker()
    session_state = worker_state(sessionId)

    handle = Handle(streams_per_handle)
    nccl_comm = session_state["nccl"]
    eps = session_state["ucp_eps"]
    nWorkers = session_state["nworkers"]
    workerId = session_state["wid"]

    if (verbose):
        get_worker().log_event(topic="info", msg="Injecting comms on handle.")
    inject_comms_on_handle(handle, nccl_comm, ucp_worker, eps,
                           nWorkers, workerId, verbose)

    if (verbose):
        get_worker().log_event(topic="info", msg="Finished injecting comms on handle.")

    worker_state(sessionId)["handle"] = handle


def _func_build_handle(sessionId, streams_per_handle, verbose):
    """
    Builds a handle_t on the current worker given the initialized comms

    Parameters
    ----------
    sessionId : str id to reference state for current comms instance.
    streams_per_handle : int number of internal streams to create
    verbose : bool print verbose logging output
    """
    if (verbose):
        get_worker().log_event(topic="info", msg="Finished injecting comms on handle.")

    handle = Handle(streams_per_handle)

    session_state = worker_state(sessionId)

    workerId = session_state["wid"]
    nWorkers = session_state["nworkers"]

    nccl_comm = session_state["nccl"]
    inject_comms_on_handle_coll_only(handle, nccl_comm, nWorkers,
                                     workerId, verbose)
    session_state["handle"] = handle


def _func_store_initial_state(nworkers, sessionId, uniqueId, wid):
    # TODO: We don't ever remove wid or nworkers... could cause problems? Maybe we should just blow away whole session
    session_state = worker_state(sessionId)
    session_state["nccl_uid"] = uniqueId
    session_state["wid"] = wid
    session_state["nworkers"] = nworkers


async def _func_ucp_create_endpoints(sessionId, worker_info):
    """
    Runs on each worker to create ucp endpoints to all other workers

    Parameters
    ----------
    sessionId : str
                uuid unique id for this instance
    worker_info : dict
                  Maps worker addresses to NCCL ranks & UCX ports
    """
    eps = [None] * len(worker_info)
    count = 1

    for k in worker_info:
        ip, port = parse_host_port(k)

        ep = await get_ucx().get_endpoint(ip, worker_info[k]["port"])

        eps[worker_info[k]["rank"]] = ep
        count += 1

    worker_state(sessionId)["ucp_eps"] = eps


async def _func_destroy_all(sessionId, comms_p2p, verbose=False):
    if(verbose):
        get_worker().log_event(topic="info", msg="Destroying NCCL session state.")
    session_state = worker_state(sessionId)
    if ('nccl' in session_state):
        session_state["nccl"].destroy()
        del session_state["nccl"]
        if (verbose):
            get_worker().log_event(topic="info", msg="NCCL session state destroyed.")
    else:
        if (verbose):
            get_worker().log_event(topic="warning",
                                   msg=f"Session state for, '{sessionId}', does not contain expected 'nccl' element")

    if (verbose):
        get_worker().log_event(topic="info", msg=f"Destroy CUDA handle for sessionId, '{sessionId}.'")
    if ('handle' in session_state):
        del session_state["handle"]
    else:
        if (verbose):
            get_worker().log_event(topic="warning",
                                   msg=f"Session state for, '{sessionId}', does not contain expected 'handle' element")


def _func_ucp_ports(client, workers):
    return client.run(_func_ucp_listener_port,
                      workers=workers)


def _func_worker_ranks(workers):
    """
    Builds a dictionary of { (worker_address, worker_port) : worker_rank }
    """
    return dict(list(zip(workers, range(len(workers)))))
