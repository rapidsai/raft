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


# We're still using cython v0.29.x - which doesn't have std::optional
# support. Include the minimal definition here as suggested by
# https://github.com/cython/cython/issues/3293#issuecomment-1223058101

cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass optional[T]:
        optional()
        optional& operator=[U](U&)
