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

#include "topk_for_cagra/topk_core.cuh"

namespace raft::neighbors::cagra::detail {
namespace single_cta_search {

template <unsigned MAX_INTERNAL_TOPK>
struct topk_by_radix_sort_base {
  static constexpr std::uint32_t smem_size        = MAX_INTERNAL_TOPK * 2 + 2048 + 8;
  static constexpr std::uint32_t state_bit_lenght = 0;
  static constexpr std::uint32_t vecLen           = 2;  // TODO
};
template <unsigned MAX_INTERNAL_TOPK, class IdxT, class = void>
struct topk_by_radix_sort : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {};

template <unsigned MAX_INTERNAL_TOPK, class IdxT>
struct topk_by_radix_sort<MAX_INTERNAL_TOPK, IdxT, std::enable_if_t<((MAX_INTERNAL_TOPK <= 64))>>
  : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {
  __device__ void operator()(uint32_t topk,
                             uint32_t batch_size,
                             uint32_t len_x,
                             const uint32_t* _x,
                             const IdxT* _in_vals,
                             uint32_t* _y,
                             IdxT* _out_vals,
                             uint32_t* work,
                             uint32_t* _hints,
                             bool sort,
                             uint32_t* _smem)
  {
    std::uint8_t* const state = reinterpret_cast<std::uint8_t*>(work);
    topk_cta_11_core<topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_lenght,
                     topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen,
                     64,
                     32,
                     IdxT>(topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
  }
};

#define TOP_FUNC_PARTIAL_SPECIALIZATION(V)                                           \
  template <unsigned MAX_INTERNAL_TOPK, class IdxT>                                  \
  struct topk_by_radix_sort<                                                         \
    MAX_INTERNAL_TOPK,                                                               \
    IdxT,                                                                            \
    std::enable_if_t<((MAX_INTERNAL_TOPK <= V) && (2 * MAX_INTERNAL_TOPK > V))>>     \
    : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {                                   \
    __device__ void operator()(uint32_t topk,                                        \
                               uint32_t batch_size,                                  \
                               uint32_t len_x,                                       \
                               const uint32_t* _x,                                   \
                               const IdxT* _in_vals,                                 \
                               uint32_t* _y,                                         \
                               IdxT* _out_vals,                                      \
                               uint32_t* work,                                       \
                               uint32_t* _hints,                                     \
                               bool sort,                                            \
                               uint32_t* _smem)                                      \
    {                                                                                \
      assert(blockDim.x >= V / 4);                                                   \
      std::uint8_t* state = (std::uint8_t*)work;                                     \
      topk_cta_11_core<topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_lenght, \
                       topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen,           \
                       V,                                                            \
                       V / 4,                                                        \
                       IdxT>(                                                        \
        topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);       \
    }                                                                                \
  };
TOP_FUNC_PARTIAL_SPECIALIZATION(128);
TOP_FUNC_PARTIAL_SPECIALIZATION(256);
TOP_FUNC_PARTIAL_SPECIALIZATION(512);
TOP_FUNC_PARTIAL_SPECIALIZATION(1024);

}  // namespace single_cta_search
}  // namespace raft::neighbors::cagra::detail
