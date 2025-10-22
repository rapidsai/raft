# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from dask.distributed import default_client


def get_client(client=None):
    return default_client() if client is None else client


def parse_host_port(address):
    """
    Given a string address with host/port, build a tuple(host, port)

    Parameters
    ----------
    address: string address to parse

    Returns
    -------
    tuple with host and port info : tuple(host, port)
    """
    if "://" in address:
        address = address.rsplit("://", 1)[1]
    host, port = address.split(":")
    port = int(port)
    return host, port
