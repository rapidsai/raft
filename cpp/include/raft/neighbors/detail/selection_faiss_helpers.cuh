/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#pragma once

namespace raft::neighbors::detail {

// This function is used in cpp/test/neighbors/select.cu. We want to make it
// available through both the selection_faiss-inl.cuh and
// selection_faiss-ext.cuh headers.
template <typename payload_t, typename key_t>
constexpr int kFaissMaxK()
{
  if (sizeof(key_t) >= 8) { return sizeof(payload_t) >= 8 ? 512 : 1024; }
  return 2048;
}

}  // namespace raft::neighbors::detail
