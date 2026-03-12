#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# TODO(jameslamb): revert before merging
source ci/use_wheels_from_prs.sh

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRAFT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBRAFT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibraft --stable --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBRAFT_WHEELHOUSE}"/libraft*.whl \
    "$(echo "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl)[test]"

python -m pytest ./python/pylibraft/pylibraft/tests
