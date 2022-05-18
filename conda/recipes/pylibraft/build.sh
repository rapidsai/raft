# Copyright (c) 2022, NVIDIA CORPORATION.
#!/usr/bin/env bash

# This assumes the script is executed from the root of the repo directory
./build.sh pylibraft --install --no-nvtx --cmake-args=\"-DFIND_RAFT_CPP=ON\"
