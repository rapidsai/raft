/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <raft/core/c_api.h>
#include <stdlib.h>

int main()
{
  raftResources_t res;
  raftError_t create_error = raftCreateResources(&res);
  if (create_error == RAFT_ERROR) { exit(EXIT_FAILURE); }

  raftError_t destroy_error = raftDestroyResources(res);
  if (destroy_error == RAFT_ERROR) { exit(EXIT_FAILURE); }

  return 0;
}
