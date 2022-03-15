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

function(find_and_configure_mdspan VERSION)
  rapids_cpm_find(
    mdspan ${VERSION}
    GLOBAL_TARGETS std::mdspan
    BUILD_EXPORT_SET    raft-exports
    INSTALL_EXPORT_SET  raft-exports
    CPM_ARGS
      EXCLUDE_FROM_ALL TRUE
      GIT_REPOSITORY https://github.com/rapidsai/mdspan.git
      GIT_TAG b3042485358d2ee168ae2b486c98c2c61ec5aec1
      OPTIONS "MDSPAN_ENABLE_CUDA ON"
              "MDSPAN_CXX_STANDARD ON"
  )
endfunction()

find_and_configure_mdspan(0.2.0)
