# Copyright (c) 2022-2023, NVIDIA CORPORATION.

import os

import pytest

from dask_cuda import LocalCUDACluster

from raft_dask.common.utils import create_client

os.environ["UCX_LOG_LEVEL"] = "error"


@pytest.fixture(scope="session")
def cluster():
    scheduler_file = os.environ.get("SCHEDULER_FILE")
    if scheduler_file:
        yield scheduler_file
    else:
        cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
        yield cluster
        cluster.close()


@pytest.fixture(scope="session")
def ucx_cluster():
    scheduler_file = os.environ.get("SCHEDULER_FILE")
    if scheduler_file:
        yield scheduler_file
    else:
        cluster = LocalCUDACluster(
            protocol="ucx",
        )
        yield cluster
        cluster.close()


@pytest.fixture(scope="session")
def client(cluster):
    client = create_client(cluster)
    yield client
    client.close()


@pytest.fixture()
def ucx_client(ucx_cluster):
    client = create_client(ucx_cluster)
    yield client
    client.close()
