# Copyright (c) 2022-2023, NVIDIA CORPORATION.

import os

import pytest

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

os.environ["UCX_LOG_LEVEL"] = "error"


@pytest.fixture(scope="session")
def cluster():
    scheduler_file = os.environ.get("SCHEDULER_FILE")
    if scheduler_file:
        return scheduler_file
    else:
        cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
        yield cluster
        cluster.close()


@pytest.fixture(scope="session")
def ucx_cluster():
    scheduler_file = os.environ.get("SCHEDULER_FILE")
    if scheduler_file:
        return scheduler_file
    else:
        cluster = LocalCUDACluster(
            protocol="ucx",
        )
        yield cluster
        cluster.close()


@pytest.fixture(scope="session")
def client(cluster):
    if isinstance(cluster, LocalCUDACluster):
        client = Client(cluster)
    else:
        client = Client(scheduler_file=cluster)
    yield client
    client.close()


@pytest.fixture()
def ucx_client(ucx_cluster):
    if isinstance(ucx_cluster, LocalCUDACluster):
        client = Client(ucx_cluster)
    else:
        client = Client(scheduler_file=cluster)
    yield client
    client.close()
