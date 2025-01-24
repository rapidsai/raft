# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import os

import pytest

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

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
def ucxx_cluster():
    pytest.importorskip("distributed_ucxx")

    scheduler_file = os.environ.get("SCHEDULER_FILE")
    if scheduler_file:
        yield scheduler_file
    else:
        cluster = LocalCUDACluster(
            protocol="ucxx",
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


@pytest.fixture()
def ucxx_client(ucxx_cluster):
    client = create_client(ucxx_cluster)
    yield client
    client.close()


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


def pytest_addoption(parser):
    group = parser.getgroup("Dask RAFT Custom Options")

    group.addoption(
        "--run_ucx", action="store_true", help="run _only_ UCX-Py tests"
    )

    group.addoption(
        "--run_ucxx", action="store_true", help="run _only_ UCXX tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_ucx"):
        skip_others = pytest.mark.skip(
            reason="only runs when --run_ucx is not specified"
        )
        for item in items:
            if "ucx" not in item.keywords:
                item.add_marker(skip_others)
    else:
        skip_ucx = pytest.mark.skip(reason="requires --run_ucx to run")
        for item in items:
            if "ucx" in item.keywords:
                item.add_marker(skip_ucx)

    if config.getoption("--run_ucxx"):
        skip_others = pytest.mark.skip(
            reason="only runs when --run_ucxx is not specified"
        )
        for item in items:
            if "ucxx" not in item.keywords:
                item.add_marker(skip_others)
    else:
        skip_ucxx = pytest.mark.skip(reason="requires --run_ucxx to run")
        for item in items:
            if "ucxx" in item.keywords:
                item.add_marker(skip_ucxx)
