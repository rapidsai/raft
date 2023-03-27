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
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <raft/neighbors/detail/cagra/search_common.hpp>
#include <raft/neighbors/detail/cagra/search_core.h>
#include <string>

#include <raft/neighbors/detail/cagra/cagra.hpp>

namespace raft::neighbors::experimental::cagra::detail {

void create_plan_dispatch(void** plan,
                          const std::string dtype_name,
                          const std::size_t team_size,
                          const std::string search_mode,
                          const std::size_t topk,
                          const std::size_t itopk_size,
                          const std::size_t num_parents,
                          const std::size_t min_iterations,
                          const std::size_t max_iterations,
                          const std::size_t max_queries,
                          const std::size_t load_bit_length,
                          const std::size_t thread_block_size,
                          const std::string hashmap_mode,
                          const std::size_t hashmap_min_bitlen,
                          const float hashmap_max_fill_rate,
                          const std::size_t dataset_size,
                          const std::size_t dataset_dim,
                          const std::size_t graph_degree,
                          const void* dev_dataset_ptr,  // device ptr, [dataset_size, dataset_dim]
                          const INDEX_T* dev_graph_ptr  // device ptr, [dataset_size, graph_degree]
)
{
#define _SET_CREATE_FUNC_128D(DTYPE)                                            \
  unsigned _team_size = team_size;                                              \
  if (_team_size == 0) _team_size = 8;                                          \
  if (_team_size == 4) {                                                        \
    _create_plan = create_plan<DTYPE, 128, 4>;                                  \
  } else if (_team_size == 8) {                                                 \
    _create_plan = create_plan<DTYPE, 128, 8>;                                  \
  } else if (_team_size == 16) {                                                \
    _create_plan = create_plan<DTYPE, 128, 16>;                                 \
  } else if (_team_size == 32) {                                                \
    _create_plan = create_plan<DTYPE, 128, 32>;                                 \
  } else {                                                                      \
    fprintf(stderr,                                                             \
            "[CAGRA Error]\nUn-supported team size (%u)."                       \
            "The supported team sizes for this dataset are 4, 8, 16 and 32.\n", \
            _team_size);                                                        \
    exit(-1);                                                                   \
  }
#define _SET_CREATE_FUNC_256D(DTYPE)                                         \
  unsigned _team_size = team_size;                                           \
  if (_team_size == 0) _team_size = 16;                                      \
  if (_team_size == 8) {                                                     \
    _create_plan = create_plan<DTYPE, 256, 8>;                               \
  } else if (_team_size == 16) {                                             \
    _create_plan = create_plan<DTYPE, 256, 16>;                              \
  } else if (_team_size == 32) {                                             \
    _create_plan = create_plan<DTYPE, 256, 32>;                              \
  } else {                                                                   \
    fprintf(stderr,                                                          \
            "[CAGRA Error]\nUn-supported team size (%u)."                    \
            "The supported team sizes for this dataset are 8, 16 and 32.\n", \
            _team_size);                                                     \
    exit(-1);                                                                \
  }
#define _SET_CREATE_FUNC_512D(DTYPE)                                      \
  unsigned _team_size = team_size;                                        \
  if (_team_size == 0) _team_size = 32;                                   \
  if (_team_size == 16) {                                                 \
    _create_plan = create_plan<DTYPE, 512, 16>;                           \
  } else if (_team_size == 32) {                                          \
    _create_plan = create_plan<DTYPE, 512, 32>;                           \
  } else {                                                                \
    fprintf(stderr,                                                       \
            "[CAGRA Error]\nUn-supported team size (%u)."                 \
            "The supported team sizes for this dataset are 16 and 32.\n", \
            _team_size);                                                  \
    exit(-1);                                                             \
  }
#define _SET_CREATE_FUNC_1024D(DTYPE)                             \
  unsigned _team_size = team_size;                                \
  if (_team_size == 0) _team_size = 32;                           \
  if (_team_size == 32) {                                         \
    _create_plan = create_plan<DTYPE, 1024, 32>;                  \
  } else {                                                        \
    fprintf(stderr,                                               \
            "[CAGRA Error]\nUn-supported team size (%u)."         \
            "The supported team sizes for this dataset is 32.\n", \
            _team_size);                                          \
    exit(-1);                                                     \
  }
#define _SET_CREATE_FUNC(DTYPE)                                                            \
  if (dataset_dim <= 128) {                                                                \
    _SET_CREATE_FUNC_128D(DTYPE)                                                           \
  } else if (dataset_dim <= 256) {                                                         \
    _SET_CREATE_FUNC_256D(DTYPE)                                                           \
  } else if (dataset_dim <= 512) {                                                         \
    _SET_CREATE_FUNC_512D(DTYPE)                                                           \
  } else if (dataset_dim <= 1024) {                                                        \
    _SET_CREATE_FUNC_1024D(DTYPE)                                                          \
  } else {                                                                                 \
    fprintf(stderr, "[CAGRA Error]\nDataset dimension is too large (%lu)\n", dataset_dim); \
    exit(-1);                                                                              \
  }
#define SET_CREATE_FUNC() \
  if (dtype_name == "float") { _SET_CREATE_FUNC(float); }
  /* else if (dtype_name == "half") {  \
     _SET_CREATE_FUNC(half);           \
   } else if (dtype_name == "int8") {  \
     _SET_CREATE_FUNC(int8_t);         \
   } else if (dtype_name == "uint8") { \
     _SET_CREATE_FUNC(uint8_t);        \
   }*/

  typedef void (*create_plan_t)(void** plan,
                                const std::string search_mode,
                                const std::size_t topk,
                                const std::size_t itopk_size,
                                const std::size_t num_parents,
                                const std::size_t min_iterations,
                                const std::size_t max_iterations,
                                const std::size_t max_queries,
                                const std::size_t load_bit_length,
                                const std::size_t thread_block_size,
                                const std::string hashmap_mode,
                                const std::size_t hashmap_min_bitlen,
                                const float hashmap_max_fill_rate,
                                const std::size_t dataset_size,
                                const std::size_t dataset_dim,
                                const std::size_t graph_degree,
                                const void* dev_dataset_ptr,
                                const INDEX_T* dev_graph_ptr);
  create_plan_t _create_plan;
  SET_CREATE_FUNC();
  _create_plan(plan,
               search_mode,
               topk,
               itopk_size,
               num_parents,
               min_iterations,
               max_iterations,
               max_queries,
               load_bit_length,
               thread_block_size,
               hashmap_mode,
               hashmap_min_bitlen,
               hashmap_max_fill_rate,
               dataset_size,
               dataset_dim,
               graph_degree,
               dev_dataset_ptr,
               dev_graph_ptr);
}

//
void search_dispatch(void* plan,
                     INDEX_T* dev_topk_indices_ptr,       // [num_queries, topk]
                     DISTANCE_T* dev_topk_distances_ptr,  // [num_queries, topk]
                     const void* dev_query_ptr,           // [num_queries, query_dim]
                     const uint32_t num_queries,
                     const uint32_t num_random_samplings,
                     const uint64_t rand_xor_mask,
                     const INDEX_T* dev_seed_ptr,  // [num_queries, num_seeds]
                     const uint32_t num_seeds,
                     uint32_t* num_executed_iterations,
                     cudaStream_t cuda_stream)
{
#define _SET_SEARCH_FUNC_128D(DTYPE)                                            \
  if (_plan->_team_size == 4) {                                                 \
    _search = search<DTYPE, 128, 4>;                                            \
  } else if (_plan->_team_size == 8) {                                          \
    _search = search<DTYPE, 128, 8>;                                            \
  } else if (_plan->_team_size == 16) {                                         \
    _search = search<DTYPE, 128, 16>;                                           \
  } else if (_plan->_team_size == 32) {                                         \
    _search = search<DTYPE, 128, 32>;                                           \
  } else {                                                                      \
    fprintf(stderr,                                                             \
            "[CAGRA Error]\nUn-supported team size (%u)."                       \
            "The supported team sizes for this dataset are 4, 8, 16 and 32.\n", \
            _plan->_team_size);                                                 \
    exit(-1);                                                                   \
  }
#define _SET_SEARCH_FUNC_256D(DTYPE)                                         \
  if (_plan->_team_size == 8) {                                              \
    _search = search<DTYPE, 256, 8>;                                         \
  } else if (_plan->_team_size == 16) {                                      \
    _search = search<DTYPE, 256, 16>;                                        \
  } else if (_plan->_team_size == 32) {                                      \
    _search = search<DTYPE, 256, 32>;                                        \
  } else {                                                                   \
    fprintf(stderr,                                                          \
            "[CAGRA Error]\nUn-supported team size (%u)."                    \
            "The supported team sizes for this dataset are 8, 16 and 32.\n", \
            _plan->_team_size);                                              \
    exit(-1);                                                                \
  }
#define _SET_SEARCH_FUNC_512D(DTYPE)                                      \
  if (_plan->_team_size == 16) {                                          \
    _search = search<DTYPE, 512, 16>;                                     \
  } else if (_plan->_team_size == 32) {                                   \
    _search = search<DTYPE, 512, 32>;                                     \
  } else {                                                                \
    fprintf(stderr,                                                       \
            "[CAGRA Error]\nUn-supported team size (%u)."                 \
            "The supported team sizes for this dataset are 16 and 32.\n", \
            _plan->_team_size);                                           \
    exit(-1);                                                             \
  }
#define _SET_SEARCH_FUNC_1024D(DTYPE)                             \
  if (_plan->_team_size == 32) {                                  \
    _search = search<DTYPE, 1024, 32>;                            \
  } else {                                                        \
    fprintf(stderr,                                               \
            "[CAGRA Error]\nUn-supported team size (%u)."         \
            "The supported team sizes for this dataset is 32.\n", \
            _plan->_team_size);                                   \
    exit(-1);                                                     \
  }
#define _SET_SEARCH_FUNC(DTYPE)                                                                 \
  if (_plan->_max_dataset_dim <= 128) {                                                         \
    _SET_SEARCH_FUNC_128D(DTYPE)                                                                \
  } else if (_plan->_max_dataset_dim <= 256) {                                                  \
    _SET_SEARCH_FUNC_256D(DTYPE)                                                                \
  } else if (_plan->_max_dataset_dim <= 512) {                                                  \
    _SET_SEARCH_FUNC_512D(DTYPE)                                                                \
  } else if (_plan->_max_dataset_dim <= 1024) {                                                 \
    _SET_SEARCH_FUNC_1024D(DTYPE)                                                               \
  } else {                                                                                      \
    fprintf(                                                                                    \
      stderr, "[CAGRA Error]\nDataset dimension is too large (%u)\n", _plan->_max_dataset_dim); \
    exit(-1);                                                                                   \
  }
#define SET_SEARCH_FUNC() \
  if (_plan->_dtype == CUDA_R_32F) { _SET_SEARCH_FUNC(float); }
  /* else if (_plan->_dtype == CUDA_R_16F) { \
     _SET_SEARCH_FUNC(half);                 \
   } else if (_plan->_dtype == CUDA_R_8I) {  \
     _SET_SEARCH_FUNC(int8_t);               \
   } else if (_plan->_dtype == CUDA_R_8U) {  \
     _SET_SEARCH_FUNC(uint8_t);              \
   }*/

  search_common* _plan = (search_common*)plan;
  typedef void (*search_t)(void* plan,
                           INDEX_T* dev_topk_indices_ptr,
                           DISTANCE_T* dev_topk_distances_ptr,
                           const void* dev_query_ptr,
                           const uint32_t num_queries,
                           const uint32_t num_random_samplings,
                           const uint64_t rand_xor_mask,
                           const INDEX_T* dev_seed_ptr,
                           const uint32_t num_seeds,
                           uint32_t* num_executed_iterations,
                           cudaStream_t cuda_stream);
  search_t _search;
  SET_SEARCH_FUNC();
  _search(plan,
          dev_topk_indices_ptr,
          dev_topk_distances_ptr,
          dev_query_ptr,
          num_queries,
          num_random_samplings,
          rand_xor_mask,
          dev_seed_ptr,
          num_seeds,
          num_executed_iterations,
          cuda_stream);
}

//
void destroy_plan_dispatch(void* plan)
{
#define _SET_DESTROY_FUNC_128D(DTYPE)                                           \
  if (_plan->_team_size == 4) {                                                 \
    _destroy_plan = destroy_plan<DTYPE, 128, 4>;                                \
  } else if (_plan->_team_size == 8) {                                          \
    _destroy_plan = destroy_plan<DTYPE, 128, 8>;                                \
  } else if (_plan->_team_size == 16) {                                         \
    _destroy_plan = destroy_plan<DTYPE, 128, 16>;                               \
  } else if (_plan->_team_size == 32) {                                         \
    _destroy_plan = destroy_plan<DTYPE, 128, 32>;                               \
  } else {                                                                      \
    fprintf(stderr,                                                             \
            "[CAGRA Error]\nUn-supported team size (%u)."                       \
            "The supported team sizes for this dataset are 4, 8, 16 and 32.\n", \
            _plan->_team_size);                                                 \
    exit(-1);                                                                   \
  }
#define _SET_DESTROY_FUNC_256D(DTYPE)                                        \
  if (_plan->_team_size == 8) {                                              \
    _destroy_plan = destroy_plan<DTYPE, 256, 8>;                             \
  } else if (_plan->_team_size == 16) {                                      \
    _destroy_plan = destroy_plan<DTYPE, 256, 16>;                            \
  } else if (_plan->_team_size == 32) {                                      \
    _destroy_plan = destroy_plan<DTYPE, 256, 32>;                            \
  } else {                                                                   \
    fprintf(stderr,                                                          \
            "[CAGRA Error]\nUn-supported team size (%u)."                    \
            "The supported team sizes for this dataset are 8, 16 and 32.\n", \
            _plan->_team_size);                                              \
    exit(-1);                                                                \
  }
#define _SET_DESTROY_FUNC_512D(DTYPE)                                     \
  if (_plan->_team_size == 16) {                                          \
    _destroy_plan = destroy_plan<DTYPE, 512, 16>;                         \
  } else if (_plan->_team_size == 32) {                                   \
    _destroy_plan = destroy_plan<DTYPE, 512, 32>;                         \
  } else {                                                                \
    fprintf(stderr,                                                       \
            "[CAGRA Error]\nUn-supported team size (%u)."                 \
            "The supported team sizes for this dataset are 16 and 32.\n", \
            _plan->_team_size);                                           \
    exit(-1);                                                             \
  }
#define _SET_DESTROY_FUNC_1024D(DTYPE)                            \
  if (_plan->_team_size == 32) {                                  \
    _destroy_plan = destroy_plan<DTYPE, 1024, 32>;                \
  } else {                                                        \
    fprintf(stderr,                                               \
            "[CAGRA Error]\nUn-supported team size (%u)."         \
            "The supported team sizes for this dataset is 32.\n", \
            _plan->_team_size);                                   \
    exit(-1);                                                     \
  }
#define _SET_DESTROY_FUNC(DTYPE)                                                                \
  if (_plan->_max_dataset_dim <= 128) {                                                         \
    _SET_DESTROY_FUNC_128D(DTYPE)                                                               \
  } else if (_plan->_max_dataset_dim <= 256) {                                                  \
    _SET_DESTROY_FUNC_256D(DTYPE)                                                               \
  } else if (_plan->_max_dataset_dim <= 512) {                                                  \
    _SET_DESTROY_FUNC_512D(DTYPE)                                                               \
  } else if (_plan->_max_dataset_dim <= 1024) {                                                 \
    _SET_DESTROY_FUNC_1024D(DTYPE)                                                              \
  } else {                                                                                      \
    fprintf(                                                                                    \
      stderr, "[CAGRA Error]\nDataset dimension is too large (%u)\n", _plan->_max_dataset_dim); \
    exit(-1);                                                                                   \
  }
#define SET_DESTROY_FUNC() \
  if (_plan->_dtype == CUDA_R_32F) { _SET_DESTROY_FUNC(float); }
  /*else if (_plan->_dtype == CUDA_R_16F) { \
    _SET_DESTROY_FUNC(half);                \
  } else if (_plan->_dtype == CUDA_R_8I) {  \
    _SET_DESTROY_FUNC(int8_t);              \
  } else if (_plan->_dtype == CUDA_R_8U) {  \
    _SET_DESTROY_FUNC(uint8_t);             \
  }*/

  search_common* _plan = (search_common*)plan;
  typedef void (*destroy_plan_t)(void* plan);
  destroy_plan_t _destroy_plan;
  SET_DESTROY_FUNC();
  _destroy_plan(plan);
}
}  // namespace raft::neighbors::experimental::cagra::detail