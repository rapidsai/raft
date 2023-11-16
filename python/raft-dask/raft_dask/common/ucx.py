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

import ucp


async def _connection_func(ep):
    UCX.get().add_server_endpoint(ep)


class UCX:
    """
    Singleton UCX context to encapsulate all interactions with the
    UCX-py API and guarantee only a single listener & endpoints are
    created by RAFT Comms on a single process.
    """

    __instance = None

    def __init__(self, listener_callback):

        self.listener_callback = listener_callback

        self._create_listener()
        self._endpoints = {}
        self._server_endpoints = []

        assert UCX.__instance is None

        UCX.__instance = self

    @staticmethod
    def get(listener_callback=_connection_func):
        if UCX.__instance is None:
            UCX(listener_callback)
        return UCX.__instance

    def get_worker(self):
        return ucp.get_ucp_worker()

    def _create_listener(self):
        self._listener = ucp.create_listener(self.listener_callback)

    def listener_port(self):
        return self._listener.port

    async def _create_endpoint(self, ip, port):
        ep = await ucp.create_endpoint(ip, port)
        self._endpoints[(ip, port)] = ep
        return ep

    def add_server_endpoint(self, ep):
        self._server_endpoints.append(ep)

    async def get_endpoint(self, ip, port):
        if (ip, port) not in self._endpoints:
            ep = await self._create_endpoint(ip, port)
        else:
            ep = self._endpoints[(ip, port)]

        return ep

    async def close_endpoints(self):
        for k, ep in self._endpoints.items():
            await ep.close()

        for ep in self._server_endpoints:
            ep.close()

    def __del__(self):
        for ip_port, ep in self._endpoints.items():
            if not ep.closed():
                ep.abort()
            del ep

        for ep in self._server_endpoints:
            if not ep.closed():
                ep.abort()
            del ep

        self._listener.close()
