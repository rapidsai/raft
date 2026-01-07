#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRAFT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
    PYLIBRAFT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibraft --stable --cuda "$RAPIDS_CUDA_VERSION")")
    source ./ci/use_upstream_sabi_wheels.sh
else
    PYLIBRAFT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    "${LIBRAFT_WHEELHOUSE}"/libraft*.whl \
    "$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl)[test]"

python -m pytest ./python/pylibraft/pylibraft/tests
