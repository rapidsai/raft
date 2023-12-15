# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import numpy as np

import rmm


class device_ndarray:
    """
    pylibraft.common.device_ndarray is meant to be a very lightweight
    __cuda_array_interface__ wrapper around a numpy.ndarray.
    """

    def __init__(self, np_ndarray):
        """
        Construct a pylibraft.common.device_ndarray wrapper around a
        numpy.ndarray

        Parameters
        ----------
        ndarray : Can be numpy.ndarray, array like or even directly
            an __array_interface__. Only case it is a numpy.ndarray its
            contents will be copied to the device.

        Examples
        --------
        The device_ndarray is __cuda_array_interface__ compliant so it is
        interoperable with other libraries that also support it, such as
        CuPy and PyTorch.

        The following usage example demonstrates
        converting a pylibraft.common.device_ndarray to a cupy.ndarray:
        .. code-block:: python

            import cupy as cp
            from pylibraft.common import device_ndarray

            raft_array = device_ndarray.empty((100, 50))
            cupy_array = cp.asarray(raft_array)

        And the converting pylibraft.common.device_ndarray to a PyTorch tensor:
        .. code-block:: python

            import torch
            from pylibraft.common import device_ndarray

            raft_array = device_ndarray.empty((100, 50))
            torch_tensor = torch.as_tensor(raft_array, device='cuda')
        """

        if type(np_ndarray) is np.ndarray:
            # np_ndarray IS an actual numpy.ndarray
            self.__array_interface__ = np_ndarray.__array_interface__.copy()
            self.ndarray_ = np_ndarray
            copy = True
        elif hasattr(np_ndarray, "__array_interface__"):
            # np_ndarray HAS an __array_interface__
            self.__array_interface__ = np_ndarray.__array_interface__.copy()
            self.ndarray_ = np_ndarray
            copy = False
        elif all(
            name in np_ndarray for name in {"typestr", "shape", "version"}
        ):
            # np_ndarray IS an __array_interface__
            self.__array_interface__ = np_ndarray.copy()
            self.ndarray_ = None
            copy = False
        else:
            raise ValueError(
                "np_ndarray should be or contain __array_interface__"
            )

        order = "C" if self.c_contiguous else "F"
        if copy:
            self.device_buffer_ = rmm.DeviceBuffer.to_device(
                self.ndarray_.tobytes(order=order)
            )
        else:
            self.device_buffer_ = rmm.DeviceBuffer(
                size=np.prod(self.shape) * self.dtype.itemsize
            )

    @classmethod
    def empty(cls, shape, dtype=np.float32, order="C"):
        """
        Return a new device_ndarray of given shape and type, without
        initializing entries.

        Parameters
        ----------
        shape : int or tuple of int
                Shape of the empty array, e.g., (2, 3) or 2.
        dtype : data-type, optional
                Desired output data-type for the array, e.g, numpy.int8.
                Default is numpy.float32.
        order : {'C', 'F'}, optional (default: 'C')
                Whether to store multi-dimensional dat ain row-major (C-style)
                or column-major (Fortran-style) order in memory
        """
        arr = np.empty(shape, dtype=dtype, order=order)
        return cls(arr.__array_interface__.copy())

    @property
    def c_contiguous(self):
        """
        Is the current device_ndarray laid out in row-major format?
        """
        strides = self.strides
        return strides is None or strides[1] == self.dtype.itemsize

    @property
    def f_contiguous(self):
        """
        Is the current device_ndarray laid out in column-major format?
        """
        return not self.c_contiguous

    @property
    def dtype(self):
        """
        Datatype of the current device_ndarray instance
        """
        array_interface = self.__array_interface__
        return np.dtype(array_interface["typestr"])

    @property
    def shape(self):
        """
        Shape of the current device_ndarray instance
        """
        array_interface = self.__array_interface__
        return array_interface["shape"]

    @property
    def strides(self):
        """
        Strides of the current device_ndarray instance
        """
        array_interface = self.__array_interface__
        return array_interface.get("strides")

    @property
    def __cuda_array_interface__(self):
        """
        Returns the __cuda_array_interface__ compliant dict for
        integrating with other device-enabled libraries using
        zero-copy semantics.
        """
        device_cai = self.device_buffer_.__cuda_array_interface__
        host_cai = self.__array_interface__.copy()
        host_cai["data"] = (device_cai["data"][0], device_cai["data"][1])

        return host_cai

    def copy_to_host(self):
        """
        Returns a new numpy.ndarray object on host with the current contents of
        this device_ndarray
        """
        ret = np.frombuffer(
            self.device_buffer_.tobytes(),
            dtype=self.dtype,
            like=self.ndarray_,
        ).astype(self.dtype)
        ret = np.lib.stride_tricks.as_strided(ret, self.shape, self.strides)
        return ret
