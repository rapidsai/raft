#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import os

import versioneer
from setuptools import find_packages
from skbuild import setup


def exclude_libcxx_symlink(cmake_manifest):
    return list(
        filter(
            lambda name: not ("include/rapids/libcxx/include" in name),
            cmake_manifest,
        )
    )


setup(
    name="pylibraft" + os.getenv("RAPIDS_PY_WHEEL_CUDA_SUFFIX", default=""),
    description="RAFT: Reusable Algorithms Functions and other Tools",
    version=versioneer.get_version(),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
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
                "pylibraft.distance",
                "pylibraft.distance.includes",
                "pylibraft.common",
                "pylibraft.common.includes",
                "pylibraft.random",
                "pylibraft.random.includes",
            ]
        )
    },
    install_requires=[
        "numpy",
        "cuda-python>=11.5,<11.7.1",
        f"rmm{os.getenv('RAPIDS_PY_WHEEL_CUDA_SUFFIX', default='')}",
    ],
    # Don't want libcxx getting pulled into wheel builds.
    cmake_process_manifest_hook=exclude_libcxx_symlink,
    packages=find_packages(include=["pylibraft", "pylibraft.*"]),
    license="Apache 2.0",
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
