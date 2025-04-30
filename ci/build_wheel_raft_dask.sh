#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/raft-dask"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libraft wheels from this current build,
# then ensures 'raft-dask' wheel builds always use the 'libraft' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
LIBRAFT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRAFT_WHEELHOUSE}"/libraft_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
# raft-dask CUDA 11 package still vendors libnccl.so, which is why the size is larger
if [[ "${RAPIDS_CUDA_MAJOR}" == "11" ]]; then
    export PYDISTCHECK_MAX_SIZE="300M"
else
    export PYDISTCHECK_MAX_SIZE="2M"
fi

ci/build_wheel.sh raft-dask ${package_dir} python
ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
