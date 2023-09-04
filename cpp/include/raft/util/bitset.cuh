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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace raft::utils {
namespace detail {

/**
 * @brief Unset bits in bitset already created
 *
 * @tparam IdxT
 * @param bitset
 * @param sample_index_ptr
 * @param sample_len
 */
template <typename IdxT>
__global__ void unset_kernel(uint32_t* bitset, const IdxT* sample_index_ptr, const IdxT sample_len)
{
  for (IdxT tid = threadIdx.x + blockIdx.x * blockDim.x; tid <= sample_len;
       tid += blockDim.x * gridDim.x) {
    IdxT sample_index      = sample_index_ptr[tid];
    const IdxT bit_element = sample_index / 32;
    const IdxT bit_index   = sample_index % 32;
    const uint32_t bitmask = 1 << bit_index;
    atomicAnd(bitset + bit_element, ~bitmask);
  }
}

/**
 * @brief Create bitset from list of indices to unset
 *
 * @tparam IdxT
 * @tparam TPB
 * @param bitset
 * @param bitset_size
 * @param index_ptr
 * @param index_len
 */
template <typename IdxT, int TPB>
__global__ void create_bitset_kernel(uint32_t* bitset,
                                     const IdxT bitset_size,
                                     const IdxT* index_ptr,
                                     const IdxT index_len)
{
  extern __shared__ std::uint32_t shared_mem[];

  // Create bitset in shmem
  for (IdxT tid = threadIdx.x; tid < bitset_size; tid += TPB) {
    shared_mem[tid] = 0xffffffff;
  }

  __syncthreads();

  for (IdxT tid = threadIdx.x; tid < index_len; tid += TPB) {
    const IdxT sample_index     = index_ptr[tid];
    const IdxT bit_element      = sample_index / 32;
    const IdxT bit_index        = sample_index % 32;
    const std::uint32_t bitmask = 1 << bit_index;
    atomicAnd(shared_mem + bit_element, ~bitmask);
  }

  __syncthreads();
  // Output bitset
  for (IdxT tid = threadIdx.x; tid < bitset_size; tid += TPB) {
    bitset[tid] = shared_mem[tid];
  }
}
}  // namespace detail

template <typename IdxT = uint32_t>
struct bitset_view {
  using BitsetT    = uint32_t;
  IdxT bitset_size = sizeof(BitsetT) * 8;

  _RAFT_HOST_DEVICE bitset_view(BitsetT* bitset_ptr, IdxT bitset_len)
    : bitset_ptr_{bitset_ptr}, bitset_len_{bitset_len}
  {
  }
  _RAFT_HOST_DEVICE bitset_view(raft::device_vector_view<BitsetT, IdxT> bitset_span)
    : bitset_ptr_{bitset_span.data_handle()}, bitset_len_{bitset_span.extent(0)}
  {
  }

  inline _RAFT_DEVICE bool test(const IdxT sample_index) const
  {
    const IdxT bit_element = bitset_ptr_[sample_index / bitset_size];
    const IdxT bit_index   = sample_index % bitset_size;
    const bool is_bit_set  = (bit_element & (1ULL << bit_index)) != 0;
    return is_bit_set;
  }
  inline _RAFT_HOST_DEVICE auto get_bitset_ptr() -> BitsetT* { return bitset_ptr_; }
  inline _RAFT_HOST_DEVICE auto get_bitset_ptr() const -> const BitsetT* { return bitset_ptr_; }
  inline _RAFT_HOST_DEVICE auto get_bitset_len() const -> IdxT { return bitset_len_; }

 private:
  BitsetT* bitset_ptr_;
  IdxT bitset_len_;
};

template <typename IdxT = uint32_t>
struct bitset {
  using BitsetT    = uint32_t;
  IdxT bitset_size = sizeof(BitsetT) * 8;

  bitset(const raft::resources& res,
         raft::device_vector_view<const IdxT, IdxT> mask_index,
         IdxT bitset_len)
    : bitset_{raft::make_device_vector<BitsetT, IdxT>(res, raft::ceildiv(bitset_len, bitset_size))}
  {
    static const size_t TPB_X = 128;
    dim3 blocks(raft::ceildiv(size_t(raft::ceildiv(bitset_len, bitset_size)), TPB_X));
    dim3 threads(TPB_X);

    detail::create_bitset_kernel<IdxT, 128>
      <<<blocks,
         threads,
         raft::ceildiv(bitset_len, bitset_size) * sizeof(std::uint32_t),
         resource::get_cuda_stream(res)>>>(
        bitset_.data_handle(), bitset_.extent(0), mask_index.data_handle(), mask_index.extent(0));
  }
  // Disable copy constructor
  bitset(const bitset&)            = delete;
  bitset(bitset&&)                 = default;
  bitset& operator=(const bitset&) = delete;
  bitset& operator=(bitset&&)      = default;

  inline auto view() -> bitset_view<IdxT> { return bitset_view<IdxT>(bitset_.view()); }
  [[nodiscard]] inline auto view() const -> bitset_view<IdxT>
  {
    return bitset_view<IdxT>(bitset_.view());
  }

 private:
  raft::device_vector<BitsetT, IdxT> bitset_;
};

template <typename IdxT>
void unset_bitset(const raft::resources& res,
                  bitset_view<IdxT>& bitset_view_,
                  raft::device_vector_view<const IdxT, IdxT> mask_index)
{
  static const size_t TPB_X = 128;
  dim3 blocks(raft::ceildiv(mask_index.extents(), TPB_X));
  dim3 threads(TPB_X);
  detail::unset_kernel<<<blocks, threads, 0, resource::get_cuda_stream(res)>>>(
    bitset_view_.get_bitset_ptr(), mask_index.data_handle(), mask_index.extent(0));
}
}  // namespace raft::utils