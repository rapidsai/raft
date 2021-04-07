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
 #include "bloom_filter_strategy.cuh"

 #include <cuco/static_map.cuh>

 namespace raft {
 namespace sparse {
 namespace distance {

 template <typename value_idx, typename value_t, int tpb>
 class hash_strategy : public coo_spmv_strategy<value_idx, value_t> {
  public:
   // namespace cg = cooperative_groups;
   using insert_type =
     typename cuco::static_map<value_idx, value_t,
                               cuda::thread_scope_block>::device_mutable_view;
   using smem_type = typename insert_type::slot_type *;
   using find_type =
     typename cuco::static_map<value_idx, value_t,
                               cuda::thread_scope_block>::device_view;

   hash_strategy(const distances_config_t<value_idx, value_t> &config_, float capacity_threshold_ = 0.5 )
     : coo_spmv_strategy<value_idx, value_t>(config_), capacity_threshold(capacity_threshold_) {
     this->smem = raft::getSharedMemPerBlock();
   }

   void chunking_needed(const value_idx *indptr, const value_idx n_rows,
    rmm::device_uvector<value_idx> &mask_indptr,
    std::tuple<value_idx, value_idx> &n_rows_divided, cudaStream_t stream) {
      auto policy = rmm::exec_policy(stream);

      auto less = thrust::copy_if(policy, thrust::make_counting_iterator(value_idx(0)), thrust::make_counting_iterator(n_rows), mask_indptr.data(), fits_in_hash_table(indptr, 0, capacity_threshold * map_size()));
      std::get<0>(n_rows_divided) = less - mask_indptr.data();

      auto more = thrust::copy_if(policy, thrust::make_counting_iterator(value_idx(0)), thrust::make_counting_iterator(n_rows), less, fits_in_hash_table(indptr, capacity_threshold * map_size(), std::numeric_limits<value_idx>::max()));
      std::get<1>(n_rows_divided) = more - less;
    }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch(value_t *out_dists, value_idx *coo_rows_b,
                product_f product_func, accum_f accum_func, write_f write_func,
                int chunk_size) {
    // auto need = chunking_needed(this->config.a_indptr, this->config.a_nrows);
    // std::cout << "n: " << this->config.b_nrows << std::endl;
    // raft::print_device_vector("indptr_A", this->config.a_indptr, this->config.a_nrows + 1, std::cout);

    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
    rmm::device_uvector<value_idx> mask_indptr(this->config.a_nrows, this->config.stream);
    std::tuple<value_idx, value_idx> n_rows_divided;

    chunking_needed(this->config.a_indptr, this->config.a_nrows, mask_indptr, n_rows_divided, this->config.stream);

    auto less_rows = std::get<0>(n_rows_divided);
    // std::cout << "less_rows: " << less_rows << std::endl;
    if (less_rows > 0) {
      mask_row_it<value_idx> less(this->config.a_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base(*this, this->smem, map_size(), less, out_dists, coo_rows_b,
                            product_func, accum_func, write_func, chunk_size,
                            n_less_blocks, n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    // std::cout << "more_rows: " << more_rows << std::endl;
    if (more_rows > 0) {
      chunked_mask_row_it<value_idx> more(
        this->config.a_indptr, more_rows, mask_indptr.data() + less_rows,
        capacity_threshold * map_size(), this->config.stream);
      more.init();

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base(*this, this->smem, map_size(), more, out_dists, coo_rows_b,
                            product_func, accum_func, write_func, chunk_size,
                            n_more_blocks, n_blocks_per_row);
    }

  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t *out_dists, value_idx *coo_rows_a,
                    product_f product_func, accum_f accum_func,
                    write_f write_func, int chunk_size) {

    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
    rmm::device_uvector<value_idx> mask_indptr(this->config.b_nrows, this->config.stream);
    std::tuple<value_idx, value_idx> n_rows_divided;

    chunking_needed(this->config.b_indptr, this->config.b_nrows, mask_indptr, n_rows_divided, this->config.stream);

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<value_idx> less(this->config.b_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base_rev(*this, this->smem, map_size(), less, out_dists, coo_rows_a,
                            product_func, accum_func, write_func, chunk_size,
                            n_less_blocks, n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      chunked_mask_row_it<value_idx> more(
        this->config.b_indptr, more_rows, mask_indptr.data() + less_rows,
        capacity_threshold * map_size(), this->config.stream);
      more.init();

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base_rev(*this, this->smem, map_size(), more, out_dists, coo_rows_a,
                            product_func, accum_func, write_func, chunk_size,
                            n_more_blocks, n_blocks_per_row);
    }

  }

   __device__ inline insert_type init_insert(smem_type cache,
                                             value_idx &cache_size) {
     return insert_type::make_from_uninitialized_slots(
       cooperative_groups::this_thread_block(), cache, map_size(), -1, 0);
   }

   __device__ inline void insert(insert_type cache, value_idx &key,
                                 value_t &value, int &size) {
     auto success = cache.insert(thrust::make_pair(key, value));
   }

   __device__ inline find_type init_find(smem_type cache) {
     return find_type(cache, map_size(), -1, 0);
   }

   __device__ inline value_t find(find_type cache, value_idx &key, value_idx *indices, value_t *data, value_idx start_offset, value_idx stop_offset, int &size) {
     auto a_pair = cache.find(key);

     value_t a_col = 0.0;
     if (a_pair != cache.end()) {
       a_col = a_pair->second;
     }
     return a_col;
   }

  struct fits_in_hash_table {
  public:
    fits_in_hash_table(const value_idx *indptr_, value_idx degree_l_, value_idx degree_r_) :
     indptr(indptr_), degree_l(degree_l_), degree_r(degree_r_) {}

    __host__ __device__ bool operator()(const value_idx &i) {
      auto degree = indptr[i + 1] - indptr[i];

      return degree >= degree_l && degree < degree_r;
    }

   private:
    const value_idx *indptr;
    const value_idx degree_l, degree_r;
  };

  private:
   float capacity_threshold;
   __host__ __device__ constexpr static int map_size() {
     return (48000 - ((tpb / raft::warp_size()) * sizeof(value_t))) /
            sizeof(typename insert_type::slot_type);
    //  return 2;
   }
 };

 }  // namespace distance
 }  // namespace sparse
 }  // namespace raft
