# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Set the list of Cython files to build
set(cython_modules pairwise_distance)
set(linked_libraries raft::raft raft::distance)

# Build all of the Cython targets
# TODO: Depending on how project calls interact we can't rely on variables like CMAKE_PROJECT_NAME, otherwise we could use that to get the <PROJECT-NAME>_SOURCE_DIR and it wouldn't have to be a parameter. Can we find an alternate approach here?
create_cython_modules("${cython_modules}" "${linked_libraries}" ${raft-python_SOURCE_DIR})
