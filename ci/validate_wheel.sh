#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2


cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

PYDISTCHECK_ARGS=("--inspect")
if [[ -n "${PYDISTCHECK_MAX_SIZE+x}" ]]; then
    PYDISTCHECK_ARGS+=("--max-allowed-size-compressed" "${PYDISTCHECK_MAX_SIZE}")
fi

pydistcheck \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"
