import pytest

from dask.distributed import Client

from dask_cuda import initialize
from dask_cuda import LocalCUDACluster

enable_tcp_over_ucx = True
enable_nvlink = False
enable_infiniband = False


@pytest.fixture(scope="session")
def cluster():

    print("Created Cluster")
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    yield cluster
    cluster.close()
    print("Closed cluster")


@pytest.fixture(scope="session")
def ucx_cluster():
    initialize.initialize(create_cuda_context=True,
                          enable_tcp_over_ucx=enable_tcp_over_ucx,
                          enable_nvlink=enable_nvlink,
                          enable_infiniband=enable_infiniband)
    cluster = LocalCUDACluster(protocol="ucx",
                               enable_tcp_over_ucx=enable_tcp_over_ucx,
                               enable_nvlink=enable_nvlink,
                               enable_infiniband=enable_infiniband)
    yield cluster
    cluster.close()


@pytest.fixture(scope="session")
def client(cluster):
    client = Client(cluster)
    yield client
    client.close()


@pytest.fixture()
def ucx_client(ucx_cluster):
    client = Client(cluster)
    yield client
    client.close()
