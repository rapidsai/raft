#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from setuptools import find_packages
from skbuild import setup

install_requires = [
    "numpy",
    "numba>=0.49",
    "joblib>=0.11",
    "dask-cuda==23.4.*",
    "dask>=2023.1.1",
    "ucx-py==0.31.*",
    "distributed>=2023.1.1",
    "pylibraft==23.4.*",
]

extras_require = {
    "test": [
        "pytest",
        "dask[distributed,dataframe]",
    ]
}


def exclude_libcxx_symlink(cmake_manifest):
    return list(
        filter(
            lambda name: not ("include/rapids/libcxx/include" in name),
            cmake_manifest,
        )
    )


setup(
    name="raft-dask",
    description="Reusable Accelerated Functions & Tools Dask Infrastructure",
    version="23.04.00",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    author="NVIDIA Corporation",
    include_package_data=True,
    package_data={
        # Note: A dict comprehension with an explicit copy is necessary
        # (rather than something simpler like a dict.fromkeys) because
        # otherwise every package will refer to the same list and skbuild
        # modifies it in place.
        key: ["*.hpp", "*.pxd"]
        for key in find_packages(
            include=[
                "raft_dask.common",
                "raft_dask.common.includes",
            ]
        )
    },
    install_requires=install_requires,
    extras_require=extras_require,
    cmake_process_manifest_hook=exclude_libcxx_symlink,
    packages=find_packages(include=["raft_dask", "raft_dask.*"]),
    license="Apache 2.0",
    zip_safe=False,
)
