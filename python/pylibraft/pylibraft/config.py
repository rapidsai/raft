# Copyright (c) 2022, NVIDIA CORPORATION.
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
SUPPORTED_OUTPUT_TYPES = ["torch", "cupy", "raft"]

output_as_ = "raft"  # By default, return device_ndarray from functions


def set_output_as(output):
    """
    RAFT functions which normally return outputs with memory on device will
    instead automatically convert the output to the specified output type,
    depending on availability of the requested type.

    Parameters
    ----------
    output : str or callable. str can be either
             { "raft", "cupy", or "torch" }.
             default = "raft". callable should accept
             pylibraft.common.device_ndarray
             as a single argument and return the converted type.
    """
    if output not in SUPPORTED_OUTPUT_TYPES and not callable(output):
        raise ValueError("Unsupported output option " % output)
    global output_as_
    output_as_ = output
