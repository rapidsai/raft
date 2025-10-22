#
# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from types import SimpleNamespace

from pylibraft.common.ai_wrapper import ai_wrapper


class cai_wrapper(ai_wrapper):
    """
    Simple wrapper around a CUDA array interface object to reduce
    boilerplate for extracting common information from the underlying
    dictionary.
    """

    def __init__(self, cai_arr):
        """
        Constructor accepts a CUDA array interface compliant array

        Parameters
        ----------
        cai_arr : CUDA array interface array
        """
        helper = SimpleNamespace(
            __array_interface__=cai_arr.__cuda_array_interface__
        )
        super().__init__(helper)
        self.from_cai = True


def wrap_array(array):
    try:
        return cai_wrapper(array)
    except AttributeError:
        return ai_wrapper(array)
