# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
