# Copyright (c) 2019, NVIDIA CORPORATION.
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

import os
import random
import time

from dask.distributed import default_client
from dask.distributed import wait

from asyncio import InvalidStateError

from threading import Lock


def get_visible_devices():
    """
    Return a list of the CUDA_VISIBLE_DEVICES
    :return: list[int] visible devices
    """
    # TODO: Shouldn't have to split on every call
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def get_client(client=None):
    return default_client() if client is None else client


def parse_host_port(address):
    """
    Given a string address with host/port, build a tuple(host, port)
    :param address: string address to parse
    :return: tuple(host, port)
    """
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def build_host_dict(workers):
    """
    Builds a dict to map the set of ports running on each host to
    the hostname.
    :param workers: list(tuple(host, port)) list of worker addresses
    :return: dict(host, set(port))
    """
    hosts = set(map(lambda x: parse_host_port(x), workers))
    hosts_dict = {}
    for host, port in hosts:
        if host not in hosts_dict:
            hosts_dict[host] = set([port])
        else:
            hosts_dict[host].add(port)

    return hosts_dict


def persist_across_workers(client, objects, workers=None):
    """
    Calls persist on the 'objects' ensuring they are spread
    across the workers on 'workers'.

    Parameters
    ----------
    client : dask.distributed.Client
    objects : list
        Dask distributed objects to be persisted
    workers : list or None
        List of workers across which to persist objects
        If None, then all workers attached to 'client' will be used
    """
    if workers is None:
        workers = client.has_what().keys()  # Default to all workers
    return client.persist(objects, workers={o: workers for o in objects})


def raise_exception_from_futures(futures):
    """Raises a RuntimeError if any of the futures indicates an exception"""
    errs = [f.exception() for f in futures if f.exception()]
    if errs:
        raise RuntimeError("%d of %d worker jobs failed: %s" % (
            len(errs), len(futures), ", ".join(map(str, errs))
            ))


def wait_and_raise_from_futures(futures):
    """
    Returns the collected futures after all the futures
    have finished and do not indicate any exceptions.
    """
    wait(futures)
    raise_exception_from_futures(futures)
    return futures


def raise_mg_import_exception():
    raise Exception("cuML has not been built with multiGPU support "
                    "enabled. Build with the --multigpu flag to"
                    " enable multiGPU support.")


class MultiHolderLock:
    """
    A per-process synchronization lock allowing multiple concurrent holders
    at any one time. This is used in situations where resources might be
    limited and it's important that the number of concurrent users of
    the resources are constained.

    This lock is serializable, but relies on a Python threading.Lock
    underneath to properly synchronize internal state across threads.
    Note that this lock is only intended to be used per-process and
    the underlying threading.Lock will not be serialized.
    """

    def __init__(self, n):
        """
        Initialize the lock
        :param n : integer the maximum number of concurrent holders
        """
        self.n = n
        self.current_tasks = 0
        self.lock = Lock()

    def _acquire(self, blocking=True, timeout=10):
        lock_acquired = False

        inner_lock_acquired = self.lock.acquire(blocking, timeout)

        if inner_lock_acquired and self.current_tasks < self.n - 1:
            self.current_tasks += 1
            lock_acquired = True
            self.lock.release()

        return lock_acquired

    def acquire(self, blocking=True, timeout=10):
        """
        Acquire the lock.
        :param blocking : bool will block if True
        :param timeout : a timeout (in seconds) to wait for the lock
                         before failing.
        :return : True if lock was acquired successfully, False otherwise
        """

        t = time.time()

        lock_acquired = self._acquire(blocking, timeout)

        while blocking and not lock_acquired:

            if time.time() - t > timeout:
                raise TimeoutError()

            lock_acquired = self.acquire(blocking, timeout)
            time.sleep(random.uniform(0, 0.01))

        return lock_acquired

    def __getstate__(self):
        d = self.__dict__.copy()
        if "lock" in d:
            del d["lock"]
        return d

    def __setstate__(self, d):
        d["lock"] = Lock()
        self.__dict__ = d

    def release(self, blocking=True, timeout=10):
        """
        Release a hold on the lock to allow another holder. Note that
        while Python's threading.Lock does not have options for blocking
        or timeout in release(), this lock uses a threading.Lock
        internally and so will need to acquire that lock in order
        to properly synchronize the underlying state.
        :param blocking : bool will bock if True
        :param timeout : a timeout (in seconds) to wait for the lock
                         before failing.
        :return : True if lock was released successfully, False otherwise.
        """

        if self.current_tasks == 0:
            raise InvalidStateError("Cannot release lock when no "
                                    "concurrent tasks are executing")

        lock_acquired = self.lock.acquire(blocking, timeout)
        if lock_acquired:
            self.current_tasks -= 1
            self.lock.release()
        return lock_acquired
