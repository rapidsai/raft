#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/raft-dask"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Downloads libraft wheels from this current build,
# then ensures 'raft-dask' wheel builds always use the 'libraft' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libraft_dist
echo "libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libraft_dist/libraft_*.whl)" > /tmp/constraints.txt

LIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 cpp wheel)
PYLIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 python wheel)
LIBUCXX_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact ucxx 364 cpp wheel)
PYLIBUCXX_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact ucxx 364 python wheel)
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEEL_DIR}/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRMM_WHEEL_DIR}/rmm_*.whl)" >> /tmp/constraints.txt
echo "libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBUCXX_WHEEL_DIR}/libucxx_*.whl)" >> /tmp/constraints.txt
echo "ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBUCXX_WHEEL_DIR}/ucxx_*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

ci/build_wheel.sh raft-dask ${package_dir} python
ci/validate_wheel.sh ${package_dir} final_dist raft-dask
