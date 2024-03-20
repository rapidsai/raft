/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "compute_distance.hpp"

#include <raft/util/integer_utils.hpp>

namespace raft::neighbors::cagra::detail {
template <class DATA_T_,
          class CODE_BOOK_T_,
          unsigned PQ_BITS,
          unsigned PQ_CODE_BOOK_DIM,
          unsigned DATASET_BLOCK_DIM_,
          class DISTANCE_T,
          class INDEX_T,
          unsigned TEAM_SIZE_>
struct cagra_q_dataset_descriptor_t : public dataset_descriptor_base_t<half, DISTANCE_T, INDEX_T> {
  using LOAD_T      = device::LOAD_128BIT_T;
  using DATA_T      = DATA_T_;
  using CODE_BOOK_T = CODE_BOOK_T_;
  using QUERY_T     = typename dataset_descriptor_base_t<half, DISTANCE_T, INDEX_T>::QUERY_T;
  static const std::uint32_t DATASET_BLOCK_DIM = DATASET_BLOCK_DIM_;
  static const std::uint32_t TEAM_SIZE         = TEAM_SIZE_;

  const std::uint8_t* encoded_dataset_ptr;
  const std::uint32_t encoded_dataset_dim;
  const std::uint32_t n_subspace;
  const CODE_BOOK_T* vq_code_book_ptr;
  const float vq_scale;
  const CODE_BOOK_T* pq_code_book_ptr;
  const float pq_scale;
  using dataset_descriptor_base_t<half, DISTANCE_T, INDEX_T>::size;
  using dataset_descriptor_base_t<half, DISTANCE_T, INDEX_T>::dim;

  // Set on device
  CODE_BOOK_T* smem_pq_code_book_ptr;
  static const std::uint32_t smem_buffer_size_in_byte =
    (1 << PQ_BITS) * PQ_CODE_BOOK_DIM * utils::size_of<CODE_BOOK_T>();

  __device__ void set_smem_ptr(void* const smem_ptr)
  {
    smem_pq_code_book_ptr = reinterpret_cast<CODE_BOOK_T*>(smem_ptr);

    // Copy PQ table
    if constexpr (std::is_same<CODE_BOOK_T, half>::value) {
      for (unsigned i = threadIdx.x * 2; i < (1 << PQ_BITS) * PQ_CODE_BOOK_DIM;
           i += blockDim.x * 2) {
        half2 buf2;
        buf2.x                                                   = pq_code_book_ptr[i];
        buf2.y                                                   = pq_code_book_ptr[i + 1];
        (reinterpret_cast<half2*>(smem_pq_code_book_ptr + i))[0] = buf2;
      }
    } else {
      for (unsigned i = threadIdx.x; i < (1 << PQ_BITS) * PQ_CODE_BOOK_DIM; i += blockDim.x) {
        // TODO: vectorize
        smem_pq_code_book_ptr[i] = pq_code_book_ptr[i];
      }
    }
  }

  cagra_q_dataset_descriptor_t(const std::uint8_t* encoded_dataset_ptr,
                               const std::uint32_t encoded_dataset_dim,
                               const std::uint32_t n_subspace,
                               const CODE_BOOK_T* const vq_code_book_ptr,
                               const float vq_scale,
                               const CODE_BOOK_T* const pq_code_book_ptr,
                               const float pq_scale,
                               const std::size_t size,
                               const std::uint32_t dim)
    : dataset_descriptor_base_t<half, DISTANCE_T, INDEX_T>(size, dim),
      encoded_dataset_ptr(encoded_dataset_ptr),
      encoded_dataset_dim(encoded_dataset_dim),
      n_subspace(n_subspace),
      vq_code_book_ptr(vq_code_book_ptr),
      vq_scale(vq_scale),
      pq_code_book_ptr(pq_code_book_ptr),
      pq_scale(pq_scale)
  {
    if (dim > DATASET_BLOCK_DIM) {
      RAFT_FAIL(
        "`dim` must be smaller or equal to 512. Support for larger dimension is coming soon.");
    }
  }

  __device__ DISTANCE_T compute_similarity(const QUERY_T* const query_ptr,
                                           const INDEX_T node_id,
                                           const bool valid) const
  {
    float norm = 0;
    if (valid) {
      const unsigned lane_id = threadIdx.x % TEAM_SIZE;
      const uint32_t vq_code = *(reinterpret_cast<const std::uint32_t*>(
        encoded_dataset_ptr + (static_cast<std::uint64_t>(encoded_dataset_dim) * node_id)));
      if (PQ_BITS == 8) {
        constexpr unsigned vlen  = 4;  // **** DO NOT CHANGE ****
        constexpr unsigned nelem = raft::div_rounding_up_unsafe<unsigned>(
          DATASET_BLOCK_DIM / PQ_CODE_BOOK_DIM, TEAM_SIZE * vlen);
        // Loading PQ codes
        uint32_t pq_codes[nelem];
#pragma unroll
        for (std::uint32_t e = 0; e < nelem; e++) {
          const std::uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
          if (k >= n_subspace) break;
          // Loading 4 x 8-bit PQ-codes using 32-bit load ops (from device memory)
          pq_codes[e] = *(reinterpret_cast<const std::uint32_t*>(
            encoded_dataset_ptr + (static_cast<std::uint64_t>(encoded_dataset_dim) * node_id) + 4 +
            k));
        }
        //
        if constexpr ((std::is_same<CODE_BOOK_T, half>::value) && (PQ_CODE_BOOK_DIM % 2 == 0)) {
          // **** Use half2 for distance computation ****
          half2 norm2{0, 0};
#pragma unroll
          for (std::uint32_t e = 0; e < nelem; e++) {
            const std::uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= n_subspace) break;
            // Loading VQ code-book
            raft::TxN_t<half2, vlen / 2> vq_vals[PQ_CODE_BOOK_DIM];
#pragma unroll
            for (std::uint32_t m = 0; m < PQ_CODE_BOOK_DIM; m += 1) {
              const uint32_t d = (vlen * m) + (PQ_CODE_BOOK_DIM * k);
              if (d >= dim) break;
              vq_vals[m].load(
                reinterpret_cast<const half2*>(vq_code_book_ptr + d + (dim * vq_code)), 0);
            }
            // Compute distance
            std::uint32_t pq_code = pq_codes[e];
#pragma unroll
            for (std::uint32_t v = 0; v < vlen; v++) {
              if (PQ_CODE_BOOK_DIM * (v + k) >= dim) break;
#pragma unroll
              for (std::uint32_t m = 0; m < PQ_CODE_BOOK_DIM; m += 2) {
                const std::uint32_t d1 = m + (PQ_CODE_BOOK_DIM * v);
                const std::uint32_t d  = d1 + (PQ_CODE_BOOK_DIM * k);
                // Loading query vector in smem
                half2 diff2 = (reinterpret_cast<const half2*>(
                  query_ptr))[device::swizzling<std::uint32_t, DATASET_BLOCK_DIM / 2>(d / 2)];
                // Loading PQ code book in smem
                diff2 -= *(reinterpret_cast<half2*>(
                  smem_pq_code_book_ptr + (1 << PQ_BITS) * 2 * (m / 2) + (2 * (pq_code & 0xff))));
                diff2 -= vq_vals[d1 / vlen].val.data[(d1 % vlen) / 2];
                norm2 += diff2 * diff2;
              }
              pq_code >>= 8;
            }
          }
          norm = static_cast<float>(norm2.x + norm2.y);
        } else {
          // **** Use float for distance computation ****
#pragma unroll
          for (std::uint32_t e = 0; e < nelem; e++) {
            const std::uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
            if (k >= n_subspace) break;
            // Loading VQ code-book
            raft::TxN_t<CODE_BOOK_T, vlen> vq_vals[PQ_CODE_BOOK_DIM];
#pragma unroll
            for (std::uint32_t m = 0; m < PQ_CODE_BOOK_DIM; m++) {
              const std::uint32_t d = (vlen * m) + (PQ_CODE_BOOK_DIM * k);
              if (d >= dim) break;
              // Loading 4 x 8/16-bit VQ-values using 32/64-bit load ops (from L2$ or device memory)
              vq_vals[m].load(
                reinterpret_cast<const half2*>(vq_code_book_ptr + d + (dim * vq_code)), 0);
            }
            // Compute distance
            std::uint32_t pq_code = pq_codes[e];
#pragma unroll
            for (std::uint32_t v = 0; v < vlen; v++) {
              if (PQ_CODE_BOOK_DIM * (v + k) >= dim) break;
              raft::TxN_t<CODE_BOOK_T, PQ_CODE_BOOK_DIM> pq_vals;
              pq_vals.load(reinterpret_cast<const half2*>(smem_pq_code_book_ptr +
                                                          PQ_CODE_BOOK_DIM * (pq_code & 0xff)),
                           0);  // (from L1$ or smem)
#pragma unroll
              for (std::uint32_t m = 0; m < PQ_CODE_BOOK_DIM; m++) {
                const std::uint32_t d1 = m + (PQ_CODE_BOOK_DIM * v);
                const std::uint32_t d  = d1 + (PQ_CODE_BOOK_DIM * k);
                // if (d >= dataset_dim) break;
                DISTANCE_T diff = query_ptr[d];  // (from smem)
                diff -= pq_scale * static_cast<float>(pq_vals.data[m]);
                diff -= vq_scale * static_cast<float>(vq_vals[d1 / vlen].val.data[d1 % vlen]);
                norm += diff * diff;
              }
              pq_code >>= 8;
            }
          }
        }
      }
    }
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
      norm += __shfl_xor_sync(0xffffffff, norm, offset);
    }
    return norm;
  }
};

template <std::uint32_t DATASET_BLOCK_DIM_OUT,
          std::uint32_t TEAM_SIZE_OUT,
          std::uint32_t DATASET_BLOCK_DIM_IN,
          std::uint32_t TEAM_SIZE_IN,
          class DATA_T,
          class INDEX_T,
          class DISTANCE_T,
          class CODE_BOOK_T,
          unsigned PQ_BITS,
          unsigned PQ_CODE_BOOK_DIM>
cagra_q_dataset_descriptor_t<DATA_T,
                             CODE_BOOK_T,
                             PQ_BITS,
                             PQ_CODE_BOOK_DIM,
                             DATASET_BLOCK_DIM_OUT,
                             DISTANCE_T,
                             INDEX_T,
                             TEAM_SIZE_OUT>
set_compute_template_params(cagra_q_dataset_descriptor_t<DATA_T,
                                                         CODE_BOOK_T,
                                                         PQ_BITS,
                                                         PQ_CODE_BOOK_DIM,
                                                         DATASET_BLOCK_DIM_IN,
                                                         DISTANCE_T,
                                                         INDEX_T,
                                                         TEAM_SIZE_IN>& desc_in)
{
  return cagra_q_dataset_descriptor_t<DATA_T,
                                      CODE_BOOK_T,
                                      PQ_BITS,
                                      PQ_CODE_BOOK_DIM,
                                      DATASET_BLOCK_DIM_OUT,
                                      DISTANCE_T,
                                      INDEX_T,
                                      TEAM_SIZE_OUT>(desc_in.encoded_dataset_ptr,
                                                     desc_in.encoded_dataset_dim,
                                                     desc_in.n_subspace,
                                                     desc_in.vq_code_book_ptr,
                                                     desc_in.vq_scale,
                                                     desc_in.pq_code_book_ptr,
                                                     desc_in.pq_scale,
                                                     desc_in.size,
                                                     desc_in.dim);
}
}  // namespace raft::neighbors::cagra::detail
