# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise candidate-serialized CUDA selectors with the pinned renderer."""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RATTLER_BUILD = os.environ.get("RATTLER_BUILD") or shutil.which(
    "rattler-build"
)
RECIPE_PATHS = [
    ROOT / "conda/recipes/pylibraft/recipe.yaml",
    ROOT / "conda/recipes/raft-dask/recipe.yaml",
]


def _render(recipe_path: Path, cuda: str, platform: str, serialized: bool):
    """Render a recipe as-is or after the candidate PyYAML round trip."""
    if RATTLER_BUILD is None:
        pytest.skip("rattler-build is required for actual recipe rendering")

    environment = os.environ.copy()
    recipe_text = recipe_path.read_text()
    environment.update(
        {
            name: ""
            for name in re.findall(r'env.get\(["\']([^"\']+)', recipe_text)
        }
    )
    environment.update(
        RAPIDS_CUDA_VERSION=cuda,
        RAPIDS_PACKAGE_VERSION=(ROOT / "VERSION").read_text().strip(),
        RAPIDS_PY_VERSION="3.12",
        RAPIDS_DATETIME_STRING="261014000000",
        RAPIDS_CONDA_ARCH=platform,
        CMAKE_C_COMPILER_LAUNCHER="",
        CMAKE_CXX_COMPILER_LAUNCHER="",
        CMAKE_CUDA_COMPILER_LAUNCHER="",
        CMAKE_GENERATOR="Ninja",
        UCXX_PACKAGE_DEPENDENCY="0.48.*",
    )
    with tempfile.TemporaryDirectory(
        prefix=".candidate-selector-test-", dir=recipe_path.parent.parent
    ) as temporary_directory:
        copied_directory = Path(temporary_directory)
        shutil.copytree(
            recipe_path.parent, copied_directory, dirs_exist_ok=True
        )
        copied_recipe = copied_directory / "recipe.yaml"
        if serialized:
            copied_recipe.write_text(
                yaml.safe_dump(
                    yaml.safe_load(copied_recipe.read_text()), sort_keys=False
                )
            )
        result = subprocess.run(
            [
                RATTLER_BUILD,
                "build",
                "--recipe",
                str(copied_recipe),
                "--experimental",
                "--render-only",
                "--output-dir",
                str(copied_directory / "output"),
                "--target-platform",
                platform,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS)
@pytest.mark.parametrize("platform", ["linux-64", "linux-aarch64"])
@pytest.mark.parametrize("cuda", ["12.9.2", "13.3.0"])
@pytest.mark.parametrize("serialized", [False, True])
def test_raft_cuda_bindings_selector_survives_serialization(
    recipe_path, serialized, cuda, platform
):
    outputs = _render(recipe_path, cuda, platform, serialized)
    expected = (
        "cuda-bindings >=12.9.2,<13.0"
        if cuda.startswith("12.")
        else "cuda-bindings >=13.0.1,<14.0"
    )

    assert outputs
    for output in outputs:
        for section in ("host", "run"):
            requirements = output["recipe"]["requirements"].get(section, [])
            bindings = [
                item
                for item in requirements
                if isinstance(item, str) and item.startswith("cuda-bindings ")
            ]
            assert bindings == [expected]
