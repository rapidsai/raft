#
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
