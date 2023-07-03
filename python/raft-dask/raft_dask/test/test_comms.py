# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from collections import OrderedDict

import pytest

from dask.distributed import Client, get_worker, wait
from dask_cuda import LocalCUDACluster

try:
    from raft_dask.common import (
        Comms,
        local_handle,
        perform_test_comm_split,
        perform_test_comms_allgather,
        perform_test_comms_allreduce,
        perform_test_comms_bcast,
        perform_test_comms_device_multicast_sendrecv,
        perform_test_comms_device_send_or_recv,
        perform_test_comms_device_sendrecv,
        perform_test_comms_gather,
        perform_test_comms_gatherv,
        perform_test_comms_reduce,
        perform_test_comms_reducescatter,
        perform_test_comms_send_recv,
    )

    pytestmark = pytest.mark.mg
except ImportError:
    pytestmark = pytest.mark.skip


def create_client(cluster):
    """
    Create a Dask distributed client for a specified cluster.

    Parameters
    ----------
    cluster : LocalCUDACluster instance or str
        If a LocalCUDACluster instance is provided, a client will be created
        for it directly. If a string is provided, it should specify the path to
        a Dask scheduler file. A client will then be created for the cluster
        referenced by this scheduler file.

    Returns
    -------
    dask.distributed.Client
        A client connected to the specified cluster.
    """
    if isinstance(cluster, LocalCUDACluster):
        return Client(cluster)
    else:
        return Client(scheduler_file=cluster)


def test_comms_init_no_p2p(cluster):
    client = create_client(cluster)
    try:
        cb = Comms(verbose=True)
        cb.init()

        assert cb.nccl_initialized is True
        assert cb.ucx_initialized is False

    finally:

        cb.destroy()
        client.close()


def func_test_collective(func, sessionId, root):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return func(handle, root)


def func_test_send_recv(sessionId, n_trials):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return perform_test_comms_send_recv(handle, n_trials)


def func_test_device_send_or_recv(sessionId, n_trials):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return perform_test_comms_device_send_or_recv(handle, n_trials)


def func_test_device_sendrecv(sessionId, n_trials):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return perform_test_comms_device_sendrecv(handle, n_trials)


def func_test_device_multicast_sendrecv(sessionId, n_trials):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return perform_test_comms_device_multicast_sendrecv(handle, n_trials)


def func_test_comm_split(sessionId, n_trials):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return perform_test_comm_split(handle, n_trials)


def func_check_uid(sessionId, uniqueId, state_object):
    if not hasattr(state_object, "_raft_comm_state"):
        return 1

    state = state_object._raft_comm_state
    if sessionId not in state:
        return 2

    session_state = state[sessionId]
    if "nccl_uid" not in session_state:
        return 3

    nccl_uid = session_state["nccl_uid"]
    if nccl_uid != uniqueId:
        return 4

    return 0


def func_check_uid_on_scheduler(sessionId, uniqueId, dask_scheduler):
    return func_check_uid(
        sessionId=sessionId, uniqueId=uniqueId, state_object=dask_scheduler
    )


def func_check_uid_on_worker(sessionId, uniqueId, dask_worker=None):
    return func_check_uid(
        sessionId=sessionId, uniqueId=uniqueId, state_object=dask_worker
    )


def test_handles(cluster):
    client = create_client(cluster)

    def _has_handle(sessionId):
        return local_handle(sessionId, dask_worker=get_worker()) is not None

    try:
        cb = Comms(verbose=True)
        cb.init()

        dfs = [
            client.submit(_has_handle, cb.sessionId, pure=False, workers=[w])
            for w in cb.worker_addresses
        ]
        wait(dfs, timeout=5)

        assert all(client.compute(dfs, sync=True))

    finally:
        cb.destroy()
        client.close()


if pytestmark.markname != "skip":
    functions = [
        perform_test_comms_allgather,
        perform_test_comms_allreduce,
        perform_test_comms_bcast,
        perform_test_comms_gather,
        perform_test_comms_gatherv,
        perform_test_comms_reduce,
        perform_test_comms_reducescatter,
    ]
else:
    functions = [None]


@pytest.mark.parametrize("root_location", ["client", "worker", "scheduler"])
def test_nccl_root_placement(client, root_location):

    cb = None
    try:
        cb = Comms(
            verbose=True, client=client, nccl_root_location=root_location
        )
        cb.init()

        worker_addresses = list(
            OrderedDict.fromkeys(client.scheduler_info()["workers"].keys())
        )

        if root_location in ("worker",):
            result = client.run(
                func_check_uid_on_worker,
                cb.sessionId,
                cb.uniqueId,
                workers=[worker_addresses[0]],
            )[worker_addresses[0]]
        elif root_location in ("scheduler",):
            result = client.run_on_scheduler(
                func_check_uid_on_scheduler, cb.sessionId, cb.uniqueId
            )
        else:
            result = int(cb.uniqueId is None)

        assert result == 0

    finally:
        if cb:
            cb.destroy()


@pytest.mark.parametrize("func", functions)
@pytest.mark.parametrize("root_location", ["client", "worker", "scheduler"])
@pytest.mark.nccl
def test_collectives(client, func, root_location):

    try:
        cb = Comms(
            verbose=True, client=client, nccl_root_location=root_location
        )
        cb.init()

        for k, v in cb.worker_info(cb.worker_addresses).items():

            dfs = [
                client.submit(
                    func_test_collective,
                    func,
                    cb.sessionId,
                    v["rank"],
                    pure=False,
                    workers=[w],
                )
                for w in cb.worker_addresses
            ]
            wait(dfs, timeout=5)

            assert all([x.result() for x in dfs])
    finally:
        if cb:
            cb.destroy()


@pytest.mark.nccl
def test_comm_split(client):

    cb = Comms(comms_p2p=True, verbose=True)
    cb.init()

    dfs = [
        client.submit(
            func_test_comm_split, cb.sessionId, 3, pure=False, workers=[w]
        )
        for w in cb.worker_addresses
    ]

    wait(dfs, timeout=5)

    assert all([x.result() for x in dfs])


@pytest.mark.ucx
@pytest.mark.parametrize("n_trials", [1, 5])
def test_send_recv(n_trials, client):

    cb = Comms(comms_p2p=True, verbose=True)
    cb.init()

    dfs = [
        client.submit(
            func_test_send_recv,
            cb.sessionId,
            n_trials,
            pure=False,
            workers=[w],
        )
        for w in cb.worker_addresses
    ]

    wait(dfs, timeout=5)

    assert list(map(lambda x: x.result(), dfs))


@pytest.mark.nccl
@pytest.mark.parametrize("n_trials", [1, 5])
def test_device_send_or_recv(n_trials, client):

    cb = Comms(comms_p2p=True, verbose=True)
    cb.init()

    dfs = [
        client.submit(
            func_test_device_send_or_recv,
            cb.sessionId,
            n_trials,
            pure=False,
            workers=[w],
        )
        for w in cb.worker_addresses
    ]

    wait(dfs, timeout=5)

    assert list(map(lambda x: x.result(), dfs))


@pytest.mark.nccl
@pytest.mark.parametrize("n_trials", [1, 5])
def test_device_sendrecv(n_trials, client):

    cb = Comms(comms_p2p=True, verbose=True)
    cb.init()

    dfs = [
        client.submit(
            func_test_device_sendrecv,
            cb.sessionId,
            n_trials,
            pure=False,
            workers=[w],
        )
        for w in cb.worker_addresses
    ]

    wait(dfs, timeout=5)

    assert list(map(lambda x: x.result(), dfs))


@pytest.mark.nccl
@pytest.mark.parametrize("n_trials", [1, 5])
def test_device_multicast_sendrecv(n_trials, client):

    cb = Comms(comms_p2p=True, verbose=True)
    cb.init()

    dfs = [
        client.submit(
            func_test_device_multicast_sendrecv,
            cb.sessionId,
            n_trials,
            pure=False,
            workers=[w],
        )
        for w in cb.worker_addresses
    ]

    wait(dfs, timeout=5)

    assert list(map(lambda x: x.result(), dfs))
