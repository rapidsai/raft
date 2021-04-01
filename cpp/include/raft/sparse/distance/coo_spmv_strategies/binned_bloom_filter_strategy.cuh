/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "base_strategy.cuh"

#include <cuco/detail/hash_functions.cuh>

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx, typename value_t>
class binned_bloom_filter_strategy : public coo_spmv_strategy<value_idx, value_t> {

public:
    using smem_type = uint32_t *;
    using insert_type = smem_type;
    using find_type = smem_type;

    using Hash1 = cuco::detail::MurmurHash3_32<value_idx>;
    using Hash2 = cuco::detail::MurmurHash3_32<value_idx>;
    using Hash3 = cuco::detail::MurmurHash3_32<value_idx>;
  
    binned_bloom_filter_strategy(const distances_config_t<value_idx, value_t> &config_, mask_row_it<value_idx> &row_it_)
      : coo_spmv_strategy<value_idx, value_t>(config_),
        row_it(row_it_),
        hash1(config_.a_nnz),
        hash2(config_.a_nrows),
        hash3(config_.a_ncols) {
      this->smem = raft::getSharedMemPerBlock();
    }

    void block_divider(const value_idx *indptr, const value_idx &n_rows, rmm::device_uvector<value_idx> &mask_indptr, std::tuple<value_idx, value_idx, value_idx, value_idx> &block_counts) {
        auto policy = rmm::exec_policy(this->config.stream);
        // std::cout << "Total Rows: " << n_rows << std::endl;
        // std::cout << "Allowed nnz in 128 Block: " << nnz_in_filter<128>() << std::endl;
        auto mask_128 = thrust::copy_if(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_rows), mask_indptr.data(), nnz_filter_functor<128>(indptr));
        std::get<0>(block_counts) = mask_128 - mask_indptr.data();
        // std::cout << "128 Block Rows: " << std::get<0>(block_counts) << std::endl;

        auto mask_256 = thrust::copy_if(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_rows), mask_128, nnz_filter_functor<256>(indptr));
        std::get<1>(block_counts) = mask_256 - mask_128;
        // std::cout << "256 Block Rows: " << std::get<1>(block_counts) << std::endl;

        auto mask_512 = thrust::copy_if(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_rows), mask_256, nnz_filter_functor<512>(indptr));
        std::get<2>(block_counts) = mask_512 - mask_256;
        // std::cout << "512 Block Rows: " << std::get<2>(block_counts) << std::endl;

        auto mask_1024 = thrust::copy_if(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n_rows), mask_512, nnz_filter_functor<1024>(indptr));
        std::get<3>(block_counts) = mask_1024 - mask_512;
        // std::cout << "1024 Block Rows: " << std::get<3>(block_counts) << std::endl;
        
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch(value_t *out_dists, value_idx *coo_rows_b,
                  product_f product_func, accum_f accum_func, write_f write_func,
                  int chunk_size) {
        rmm::device_uvector<value_idx> mask_indptr(this->config.a_nrows, this->config.stream);
        auto block_counts = std::make_tuple(value_idx(0), value_idx(0), value_idx(0), value_idx(0));
        block_divider(this->config.a_indptr, this->config.a_nrows, mask_indptr, block_counts);

        // auto n_blocks = row_it.n_rows * n_blocks_per_row;

        if (std::get<0>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * 128);

            mask_row_it<value_idx> row_it_128(this->config.a_indptr, std::get<0>(block_counts), mask_indptr.data());
            this->_dispatch_base(*this, filter_bits<128>(), nnz_in_filter<128>(), row_it_128, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_128.n_rows * n_blocks_per_row, n_blocks_per_row, 128);
        }

        if (std::get<1>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * 256);

            mask_row_it<value_idx> row_it_256(this->config.a_indptr, std::get<1>(block_counts), mask_indptr.data() + std::get<0>(block_counts));
            this->_dispatch_base(*this, filter_bits<256>(), nnz_in_filter<256>(), row_it_256, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_256.n_rows * n_blocks_per_row, n_blocks_per_row, 256);
        }

        if (std::get<2>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * 512);

            mask_row_it<value_idx> row_it_512(this->config.a_indptr, std::get<2>(block_counts), mask_indptr.data() + std::get<0>(block_counts) + std::get<1>(block_counts));
            this->_dispatch_base(*this, filter_bits<512>(), nnz_in_filter<512>(), row_it_512, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_512.n_rows * n_blocks_per_row, n_blocks_per_row, 512);
        }

        if (std::get<3>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * 1024);

            mask_row_it<value_idx> row_it_1024(this->config.a_indptr, std::get<3>(block_counts), mask_indptr.data() + std::get<0>(block_counts) + std::get<1>(block_counts) + std::get<2>(block_counts));
            this->_dispatch_base(*this, filter_bits<1024>(), nnz_in_filter<1024>(), row_it_1024, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_1024.n_rows * n_blocks_per_row, n_blocks_per_row, 1024);
        }
    }

    template <typename product_f, typename accum_f, typename write_f>
    void dispatch_rev(value_t *out_dists, value_idx *coo_rows_b,
                product_f product_func, accum_f accum_func, write_f write_func,
                int chunk_size) {

        rmm::device_uvector<value_idx> mask_indptr(this->config.b_nrows, this->config.stream);
        auto block_counts = std::make_tuple(value_idx(0), value_idx(0), value_idx(0), value_idx(0));
        block_divider(this->config.b_indptr,  this->config.b_nrows, mask_indptr, block_counts);

        // auto n_blocks = row_it.n_rows * n_blocks_per_row;

        if (std::get<0>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * 128);

            mask_row_it<value_idx> row_it_128(this->config.b_indptr, std::get<0>(block_counts), mask_indptr.data());
            this->_dispatch_base_rev(*this, filter_bits<128>(), nnz_in_filter<128>(), row_it_128, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_128.n_rows * n_blocks_per_row, n_blocks_per_row);
        }

        if (std::get<1>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * 256);

            mask_row_it<value_idx> row_it_256(this->config.b_indptr, std::get<1>(block_counts), mask_indptr.data() + std::get<0>(block_counts));
            this->_dispatch_base_rev(*this, filter_bits<256>(), nnz_in_filter<256>(), row_it_256, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_256.n_rows * n_blocks_per_row, n_blocks_per_row);
        }

        if (std::get<2>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * 512);

            mask_row_it<value_idx> row_it_512(this->config.b_indptr, std::get<2>(block_counts), mask_indptr.data() + std::get<0>(block_counts) + std::get<1>(block_counts));
            this->_dispatch_base_rev(*this, filter_bits<512>(), nnz_in_filter<512>(), row_it_512, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_512.n_rows * n_blocks_per_row, n_blocks_per_row);
        }

        if (std::get<3>(block_counts) > 0) {
            auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * 1024);

            mask_row_it<value_idx> row_it_1024(this->config.b_indptr, std::get<3>(block_counts), mask_indptr.data() + std::get<0>(block_counts) + std::get<1>(block_counts) + std::get<2>(block_counts));
            this->_dispatch_base_rev(*this, filter_bits<1024>(), nnz_in_filter<1024>(), row_it_1024, out_dists, coo_rows_b,
            product_func, accum_func, write_func, chunk_size,
            row_it_1024.n_rows * n_blocks_per_row, n_blocks_per_row);
        }
    }

    __device__ inline insert_type init_insert(smem_type cache,
        value_idx &cache_size) {
        for (int k = threadIdx.x; k < cache_size; k += blockDim.x) {
            cache[k] = 0.0;
        }
        return cache;
    }

    __device__ inline void _set_key(insert_type filter, uint32_t &h) {
        constexpr auto size = sizeof(uint32_t);
        // uint32_t mem_idx = h;
        uint32_t mem_bit = size - (h % size);
        uint32_t val;
        uint32_t old;
        do {
          val = filter[h];
          old = atomicCAS(filter+h, val, val | 1 << mem_bit);
        } while(val != old);
    }

      __device__ inline void insert(insert_type filter, value_idx &key, value_t &value, int &size) {
        uint32_t hashed1 = hash1(key) & (size - 1);
        uint32_t hashed2 = hash2(key) & (size - 1);
        uint32_t hashed3 = hash3(key) & (size - 1);
        _set_key(filter, hashed1);
        _set_key(filter, hashed2);
        _set_key(filter, hashed3);
    }

    __device__ inline find_type init_find(smem_type cache) { return cache; }

    __device__ inline bool _get_key(find_type filter, uint32_t &h) {
        constexpr auto size = sizeof(uint32_t);
        // uint32_t mem_idx = h;
        uint32_t mem_bit = size - (h % size);
        return (filter[h] & 1 << mem_bit) > 0;
    }

    __device__ inline value_t find(find_type filter, value_idx &key, value_idx *indices, value_t *data, value_idx start_offset, value_idx stop_offset, int &size) {
        uint32_t hashed1 = hash1(key) & (size - 1);
        uint32_t hashed2 = hash2(key) & (size - 1);
        uint32_t hashed3 = hash3(key) & (size - 1);
        /**
         * and 2? other hash functions would be useful
         */
        auto key_present = _get_key(filter, hashed1) && _get_key(filter, hashed2) &&
                      _get_key(filter, hashed3);
        // printf("index_b: %d, key_present: %d\n", key, key_present);
        if (!key_present) {
            return 0.0;
        }
        else {
            while (start_offset <= stop_offset) {
                value_idx mid = start_offset + (stop_offset - start_offset) / 2;

                auto mid_val = indices[mid];
                if (mid_val == key) {
                    return data[mid];
                }
                else if (mid_val < key) {
                    start_offset = mid + 1;
                }
                else if (mid_val > key) {
                    stop_offset = mid - 1;
                }
            }
            return 0.0;
        }
    }

    template <int tpb>
    struct nnz_filter_functor {
      nnz_filter_functor(const value_idx *indptr_) : indptr(indptr_) {}
  
      __host__ __device__ bool operator()(const value_idx &i) {
        auto degree = indptr[i + 1] - indptr[i];
  
        if (tpb == 128) {
            // printf("Row: %d, Degree: %d, Allowed: %d, bool: %d\n", i, degree, nnz_in_filter<128>(), degree <= nnz_in_filter<128>());
            return degree <= nnz_in_filter<128>();
        }
        else if (tpb == 256) {
            return nnz_in_filter<128>() < degree && degree <= nnz_in_filter<256>();
        }
        else if (tpb == 512) {
            return nnz_in_filter<256>() < degree && degree <= nnz_in_filter<512>();
        }
        else if (tpb == 1024) {
            return degree > nnz_in_filter<512>();
        }
        else {
            return false;
        }
      }
  
     private:
      const value_idx *indptr;
    };

private:

    template <int tpb>
    __host__ __device__ constexpr static int filter_bits() {
        return ((48000 * tpb / 1024) - ((tpb / raft::warp_size()) * sizeof(value_t)));
        // return 2;
    }

    template <int tpb>
    __host__ __device__ constexpr static int nnz_in_filter() {
        constexpr auto p = 0.001;
        auto m = filter_bits<tpb>();
        constexpr auto k = 3;
        return raft::ceildiv(m, int(-k / log(1 - exp(log(p) / k))));
    }

    Hash1 hash1;
    Hash2 hash2;
    Hash3 hash3;
    mask_row_it<value_idx> &row_it;
    // static int tpb_128_nnz_filter = nnz_in_filter<128>();
    // static int tpb_256_nnz_filter = nnz_in_filter<256>();
    // static int tpb_512_nnz_filter = nnz_in_filter<512>();
    // static int tpb_1024_nnz_filter = nnz_in_filter<1024>();
};

}  // namespace distance
}  // namespace sparse
}  // namespace raft