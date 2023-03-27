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
#pragma once
#include <cuda.h>

namespace raft::neighbors::experimental::cagra::detail {

enum search_algo_t {
  SINGLE_CTA,  // for large batch
  MULTI_CTA,   // for small batch
  MULTI_KERNEL,
};

struct search_common {
  search_algo_t _algo;
  unsigned _team_size;
  unsigned _max_dataset_dim;
  cudaDataType_t _dtype;  // CUDA_R_32F, CUDA_R_16F, CUDA_R_8I, or CUDA_R_8U
  unsigned _topk;
  unsigned _max_queries;
  unsigned _dataset_dim;
};

}  // namespace raft::neighbors::experimental::cagra::detail
