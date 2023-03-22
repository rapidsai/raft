from dask.distributed import Client, get_worker, wait
from dask_cuda import LocalCUDACluster, initialize

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

import os
os.environ["UCX_LOG_LEVEL"] = "error"


def func_test_send_recv(sessionId, n_trials):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return perform_test_comms_send_recv(handle, n_trials)


def func_test_collective(func, sessionId, root):
    handle = local_handle(sessionId, dask_worker=get_worker())
    return func(handle, root)


if __name__ == "__main__":
    # initial setup
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    client = Client(cluster)

    n_trials = 5
    root_location = "client"

    # p2p test for ucx
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

    cb.destroy()

    # collectives test for nccl

    cb = Comms(
        verbose=True, client=client, nccl_root_location=root_location
    )
    cb.init()

    for k, v in cb.worker_info(cb.worker_addresses).items():

        dfs = [
            client.submit(
                func_test_collective,
                perform_test_comms_allgather,
                cb.sessionId,
                v["rank"],
                pure=False,
                workers=[w],
            )
            for w in cb.worker_addresses
        ]
        wait(dfs, timeout=5)

        assert all([x.result() for x in dfs])

    cb.destroy()

    # final client and cluster teardown
    client.close()
    cluster.close()
