#
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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


def exclude_libcxx_symlink(cmake_manifest):
    return list(
        filter(
            lambda name: not ("include/rapids/libcxx/include" in name),
            cmake_manifest,
        )
    )


setup(
    # include_package_data=True,
    # Don't want libcxx getting pulled into wheel builds.
    cmake_process_manifest_hook=exclude_libcxx_symlink,
    packages=find_packages(include=["pylibraft", "pylibraft.*"]),
    zip_safe=False,
    package_data={
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
)
