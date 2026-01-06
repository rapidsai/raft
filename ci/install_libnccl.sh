#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# TODO(jameslamb): revert this when it's pre-installed in CI images
rapids-logger "installing libnccl"
if type -f apt; then
    apt-get update
    apt-get install -y --no-install-recommends \
        libnccl-dev
else
    yum update -y
    yum install -y \
        libnccl-devel
fi
