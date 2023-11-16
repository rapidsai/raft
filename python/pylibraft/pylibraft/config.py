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
    Set output format for RAFT functions.

    Calling this function will change the output type of RAFT functions.
    By default RAFT returns a `pylibraft.common.device_ndarray` for arrays
    on GPU memory. Calling `set_output_as` allows you to have RAFT return
    arrays as cupy arrays or pytorch tensors instead. You can also have
    RAFT convert the output to other frameworks by passing a callable to
    do the conversion here.

    Notes
    -----
    Returning arrays in cupy or torch format requires you to install
    cupy or torch.

    Parameters
    ----------
    output : { "raft", "cupy", "torch" } or callable
        The output format to convert to. Can either be a str describing the
        framework to convert to, or a callable that accepts a
        device_ndarray and returns the converted type.
    """
    if output not in SUPPORTED_OUTPUT_TYPES and not callable(output):
        raise ValueError("Unsupported output option " % output)
    global output_as_
    output_as_ = output
