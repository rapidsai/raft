/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <cuco/static_map.cuh>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

// this is needed by cuco as key, value must be bitwise comparable.
// compilers don't declare float/double as bitwise comparable
// but that is too strict
// for example, the following is true (or 0):
// float a = 5;
// float b = 5;
// memcmp(&a, &b, sizeof(float));
CUCO_DECLARE_BITWISE_COMPARABLE(float);
CUCO_DECLARE_BITWISE_COMPARABLE(double);

namespace raft {
namespace sparse {
namespace distance {
namespace detail {

template <typename value_idx, typename value_t, int tpb>
class hash_strategy : public coo_spmv_strategy<value_idx, value_t, tpb> {
 public:
  using insert_type = typename cuco::legacy::
    static_map<value_idx, value_t, cuda::thread_scope_block>::device_mutable_view;
  using smem_type = typename insert_type::slot_type*;
  using find_type =
    typename cuco::legacy::static_map<value_idx, value_t, cuda::thread_scope_block>::device_view;

  hash_strategy(const distances_config_t<value_idx, value_t>& config_,
                float capacity_threshold_ = 0.5,
                int map_size_             = get_map_size())
    : coo_spmv_strategy<value_idx, value_t, tpb>(config_),
      capacity_threshold(capacity_threshold_),
      map_size(map_size_)
  {
  }

  void chunking_needed(const value_idx* indptr,
                       const value_idx n_rows,
                       rmm::device_uvector<value_idx>& mask_indptr,
                       std::tuple<value_idx, value_idx>& n_rows_divided,
                       cudaStream_t stream)
  {
    auto policy = resource::get_thrust_policy(this->config.handle);

    auto less                   = thrust::copy_if(policy,
                                thrust::make_counting_iterator(value_idx(0)),
                                thrust::make_counting_iterator(n_rows),
                                mask_indptr.data(),
                                fits_in_hash_table(indptr, 0, capacity_threshold * map_size));
    std::get<0>(n_rows_divided) = less - mask_indptr.data();

    auto more = thrust::copy_if(
      policy,
      thrust::make_counting_iterator(value_idx(0)),
      thrust::make_counting_iterator(n_rows),
      less,
      fits_in_hash_table(
        indptr, capacity_threshold * map_size, std::numeric_limits<value_idx>::max()));
    std::get<1>(n_rows_divided) = more - less;
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch(value_t* out_dists,
                value_idx* coo_rows_b,
                product_f product_func,
                accum_f accum_func,
                write_f write_func,
                int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.b_nnz, chunk_size * tpb);
    rmm::device_uvector<value_idx> mask_indptr(this->config.a_nrows,
                                               resource::get_cuda_stream(this->config.handle));
    std::tuple<value_idx, value_idx> n_rows_divided;

    chunking_needed(this->config.a_indptr,
                    this->config.a_nrows,
                    mask_indptr,
                    n_rows_divided,
                    resource::get_cuda_stream(this->config.handle));

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<value_idx> less(this->config.a_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base(*this,
                           map_size,
                           less,
                           out_dists,
                           coo_rows_b,
                           product_func,
                           accum_func,
                           write_func,
                           chunk_size,
                           n_less_blocks,
                           n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      rmm::device_uvector<value_idx> n_chunks_per_row(
        more_rows + 1, resource::get_cuda_stream(this->config.handle));
      rmm::device_uvector<value_idx> chunk_indices(0,
                                                   resource::get_cuda_stream(this->config.handle));
      chunked_mask_row_it<value_idx>::init(this->config.a_indptr,
                                           mask_indptr.data() + less_rows,
                                           more_rows,
                                           capacity_threshold * map_size,
                                           n_chunks_per_row,
                                           chunk_indices,
                                           resource::get_cuda_stream(this->config.handle));

      chunked_mask_row_it<value_idx> more(this->config.a_indptr,
                                          more_rows,
                                          mask_indptr.data() + less_rows,
                                          capacity_threshold * map_size,
                                          n_chunks_per_row.data(),
                                          chunk_indices.data(),
                                          resource::get_cuda_stream(this->config.handle));

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base(*this,
                           map_size,
                           more,
                           out_dists,
                           coo_rows_b,
                           product_func,
                           accum_func,
                           write_func,
                           chunk_size,
                           n_more_blocks,
                           n_blocks_per_row);
    }
  }

  template <typename product_f, typename accum_f, typename write_f>
  void dispatch_rev(value_t* out_dists,
                    value_idx* coo_rows_a,
                    product_f product_func,
                    accum_f accum_func,
                    write_f write_func,
                    int chunk_size)
  {
    auto n_blocks_per_row = raft::ceildiv(this->config.a_nnz, chunk_size * tpb);
    rmm::device_uvector<value_idx> mask_indptr(this->config.b_nrows,
                                               resource::get_cuda_stream(this->config.handle));
    std::tuple<value_idx, value_idx> n_rows_divided;

    chunking_needed(this->config.b_indptr,
                    this->config.b_nrows,
                    mask_indptr,
                    n_rows_divided,
                    resource::get_cuda_stream(this->config.handle));

    auto less_rows = std::get<0>(n_rows_divided);
    if (less_rows > 0) {
      mask_row_it<value_idx> less(this->config.b_indptr, less_rows, mask_indptr.data());

      auto n_less_blocks = less_rows * n_blocks_per_row;
      this->_dispatch_base_rev(*this,
                               map_size,
                               less,
                               out_dists,
                               coo_rows_a,
                               product_func,
                               accum_func,
                               write_func,
                               chunk_size,
                               n_less_blocks,
                               n_blocks_per_row);
    }

    auto more_rows = std::get<1>(n_rows_divided);
    if (more_rows > 0) {
      rmm::device_uvector<value_idx> n_chunks_per_row(
        more_rows + 1, resource::get_cuda_stream(this->config.handle));
      rmm::device_uvector<value_idx> chunk_indices(0,
                                                   resource::get_cuda_stream(this->config.handle));
      chunked_mask_row_it<value_idx>::init(this->config.b_indptr,
                                           mask_indptr.data() + less_rows,
                                           more_rows,
                                           capacity_threshold * map_size,
                                           n_chunks_per_row,
                                           chunk_indices,
                                           resource::get_cuda_stream(this->config.handle));

      chunked_mask_row_it<value_idx> more(this->config.b_indptr,
                                          more_rows,
                                          mask_indptr.data() + less_rows,
                                          capacity_threshold * map_size,
                                          n_chunks_per_row.data(),
                                          chunk_indices.data(),
                                          resource::get_cuda_stream(this->config.handle));

      auto n_more_blocks = more.total_row_blocks * n_blocks_per_row;
      this->_dispatch_base_rev(*this,
                               map_size,
                               more,
                               out_dists,
                               coo_rows_a,
                               product_func,
                               accum_func,
                               write_func,
                               chunk_size,
                               n_more_blocks,
                               n_blocks_per_row);
    }
  }

  __device__ inline insert_type init_insert(smem_type cache, const value_idx& cache_size)
  {
    return insert_type::make_from_uninitialized_slots(cooperative_groups::this_thread_block(),
                                                      cache,
                                                      cache_size,
                                                      cuco::empty_key{value_idx{-1}},
                                                      cuco::empty_value{value_t{0}});
  }

  __device__ inline void insert(insert_type cache, const value_idx& key, const value_t& value)
  {
    auto success = cache.insert(cuco::pair<value_idx, value_t>(key, value));
  }

  __device__ inline find_type init_find(smem_type cache, const value_idx& cache_size)
  {
    return find_type(
      cache, cache_size, cuco::empty_key{value_idx{-1}}, cuco::empty_value{value_t{0}});
  }

  __device__ inline value_t find(find_type cache, const value_idx& key)
  {
    auto a_pair = cache.find(key);

    value_t a_col = 0.0;
    if (a_pair != cache.end()) { a_col = a_pair->second; }
    return a_col;
  }

  struct fits_in_hash_table {
   public:
    fits_in_hash_table(const value_idx* indptr_, value_idx degree_l_, value_idx degree_r_)
      : indptr(indptr_), degree_l(degree_l_), degree_r(degree_r_)
    {
    }

    __host__ __device__ bool operator()(const value_idx& i)
    {
      auto degree = indptr[i + 1] - indptr[i];

      return degree >= degree_l && degree < degree_r;
    }

   private:
    const value_idx* indptr;
    const value_idx degree_l, degree_r;
  };

  inline static int get_map_size()
  {
    return (raft::getSharedMemPerBlock() - ((tpb / raft::warp_size()) * sizeof(value_t))) /
           sizeof(typename insert_type::slot_type);
  }

 private:
  float capacity_threshold;
  int map_size;
};

}  // namespace detail
}  // namespace distance
}  // namespace sparse
}  // namespace raft
