#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# initialize PIP_CONSTRAINT
source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX=$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")

# download wheels, store the directories holding them in variables
RMM_COMMIT=af96977c404565bbb0657c490d407a561cedc3fc
LIBRMM_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact rmm 2270 cpp wheel "${RMM_COMMIT}"
)
RMM_WHEELHOUSE=$(
  rapids-get-pr-artifact rmm 2270 python wheel --pkg_name rmm --stable "${RMM_COMMIT}"
)

UCXX_COMMIT=b1e5230d5b1ac47a11cc2d421b3af91be50b39e4
DISTRIBUTED_UCXX_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-get-pr-artifact --pkg_name distributed-ucxx ucxx 604 python wheel
)
LIBUCXX_WHEELHOUSE=$(
  RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-artifact ucxx 604 cpp wheel "${UCXX_COMMIT}"
)
UCXX_WHEELHOUSE=$(
  rapids-get-pr-artifact ucxx 604 python wheel --pkg_name ucxx --stable "${UCXX_COMMIT}"
)

# write a pip constraints file saying e.g. "whenever you encounter a requirement for 'librmm-cu12', use this wheel"
cat >> "${PIP_CONSTRAINT}" <<EOF
librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBRMM_WHEELHOUSE}"/librmm*.whl)
rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${RMM_WHEELHOUSE}"/rmm_*.whl)

distributed-ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${DISTRIBUTED_UCXX_WHEELHOUSE}"/distributed_ucxx*.whl)
libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBUCXX_WHEELHOUSE}"/libucxx*.whl)
ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${UCXX_WHEELHOUSE}"/ucxx*.whl)
EOF
