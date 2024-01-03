/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_reduce.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_helpers.cuh>
#include <raft/neighbors/ivf_pq_serialize.cuh>
#include <raft/neighbors/sample_filter.cuh>
#include <raft/random/rng.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_vector.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cub/cub.cuh>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>

namespace raft::neighbors::ivf_pq {

struct test_ivf_sample_filter {
  static constexpr unsigned offset = 1500;
};

struct ivf_pq_inputs {
  uint32_t num_db_vecs             = 4096;
  uint32_t num_queries             = 1024;
  uint32_t dim                     = 64;
  uint32_t k                       = 32;
  std::optional<double> min_recall = std::nullopt;

  ivf_pq::index_params index_params;
  ivf_pq::search_params search_params;

  // Set some default parameters for tests
  ivf_pq_inputs()
  {
    index_params.n_lists                  = max(32u, min(1024u, num_db_vecs / 128u));
    index_params.kmeans_trainset_fraction = 0.95;
  }
};

inline auto operator<<(std::ostream& os, const ivf_pq::codebook_gen& p) -> std::ostream&
{
  switch (p) {
    case ivf_pq::codebook_gen::PER_CLUSTER: os << "codebook_gen::PER_CLUSTER"; break;
    case ivf_pq::codebook_gen::PER_SUBSPACE: os << "codebook_gen::PER_SUBSPACE"; break;
    default: RAFT_FAIL("unreachable code");
  }
  return os;
}

inline auto operator<<(std::ostream& os, const ivf_pq_inputs& p) -> std::ostream&
{
  ivf_pq_inputs dflt;
  bool need_comma = false;
#define PRINT_DIFF_V(spec, val)       \
  do {                                \
    if (dflt spec != p spec) {        \
      if (need_comma) { os << ", "; } \
      os << #spec << " = " << val;    \
      need_comma = true;              \
    }                                 \
  } while (0)
#define PRINT_DIFF(spec) PRINT_DIFF_V(spec, p spec)

  os << "ivf_pq_inputs {";
  PRINT_DIFF(.num_db_vecs);
  PRINT_DIFF(.num_queries);
  PRINT_DIFF(.dim);
  PRINT_DIFF(.k);
  PRINT_DIFF_V(.min_recall, p.min_recall.value_or(0));
  PRINT_DIFF_V(.index_params.metric, print_metric{p.index_params.metric});
  PRINT_DIFF(.index_params.metric_arg);
  PRINT_DIFF(.index_params.add_data_on_build);
  PRINT_DIFF(.index_params.n_lists);
  PRINT_DIFF(.index_params.kmeans_n_iters);
  PRINT_DIFF(.index_params.kmeans_trainset_fraction);
  PRINT_DIFF(.index_params.pq_bits);
  PRINT_DIFF(.index_params.pq_dim);
  PRINT_DIFF(.index_params.codebook_kind);
  PRINT_DIFF(.index_params.force_random_rotation);
  PRINT_DIFF(.search_params.n_probes);
  PRINT_DIFF_V(.search_params.lut_dtype, print_dtype{p.search_params.lut_dtype});
  PRINT_DIFF_V(.search_params.internal_distance_dtype,
               print_dtype{p.search_params.internal_distance_dtype});
  os << "}";
  return os;
}

template <typename T>
void compare_vectors_l2(
  const raft::resources& res, T a, T b, uint32_t label, double compression_ratio, double eps)
{
  auto n_rows = a.extent(0);
  auto dim    = a.extent(1);
  rmm::mr::managed_memory_resource managed_memory;
  auto dist = make_device_mdarray<double>(res, &managed_memory, make_extents<uint32_t>(n_rows));
  linalg::map_offset(res, dist.view(), [a, b, dim] __device__(uint32_t i) {
    spatial::knn::detail::utils::mapping<float> f{};
    double d = 0.0f;
    for (uint32_t j = 0; j < dim; j++) {
      double t = f(a(i, j)) - f(b(i, j));
      d += t * t;
    }
    return sqrt(d / double(dim));
  });
  resource::sync_stream(res);
  for (uint32_t i = 0; i < n_rows; i++) {
    double d = dist(i);
    // The theoretical estimate of the error is hard to come up with,
    // the estimate below is based on experimentation + curse of dimensionality
    ASSERT_LE(d, 1.2 * eps * std::pow(2.0, compression_ratio))
      << " (label = " << label << ", ix = " << i << ", eps = " << eps << ")";
  }
}

template <typename IdxT>
auto min_output_size(const raft::resources& handle,
                     const ivf_pq::index<IdxT>& index,
                     uint32_t n_probes) -> IdxT
{
  auto acc_sizes        = index.accum_sorted_sizes();
  uint32_t last_nonzero = index.n_lists();
  while (last_nonzero > 0 && acc_sizes(last_nonzero - 1) == acc_sizes(last_nonzero)) {
    last_nonzero--;
  }
  return acc_sizes(last_nonzero) - acc_sizes(last_nonzero - std::min(last_nonzero, n_probes));
}

template <typename EvalT, typename DataT, typename IdxT>
class ivf_pq_test : public ::testing::TestWithParam<ivf_pq_inputs> {
 public:
  ivf_pq_test()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<ivf_pq_inputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);
    search_queries.resize(size_t{ps.num_queries} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    resource::sync_stream(handle_);
  }

  void calc_ref()
  {
    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    naive_knn<EvalT, DataT, IdxT>(handle_,
                                  distances_naive_dev.data(),
                                  indices_naive_dev.data(),
                                  search_queries.data(),
                                  database.data(),
                                  ps.num_queries,
                                  ps.num_db_vecs,
                                  ps.dim,
                                  ps.k,
                                  ps.index_params.metric);
    distances_ref.resize(queries_size);
    update_host(distances_ref.data(), distances_naive_dev.data(), queries_size, stream_);
    indices_ref.resize(queries_size);
    update_host(indices_ref.data(), indices_naive_dev.data(), queries_size, stream_);
    resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    auto index_view =
      raft::make_device_matrix_view<DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    return ivf_pq::build<DataT, IdxT>(handle_, ipams, index_view);
  }

  auto build_2_extends()
  {
    auto db_indices = make_device_vector<IdxT>(handle_, ps.num_db_vecs);
    linalg::map_offset(handle_, db_indices.view(), identity_op{});
    resource::sync_stream(handle_);
    auto size_1 = IdxT(ps.num_db_vecs) / 2;
    auto size_2 = IdxT(ps.num_db_vecs) - size_1;
    auto vecs_1 = database.data();
    auto vecs_2 = database.data() + size_t(size_1) * size_t(ps.dim);
    auto inds_1 = db_indices.data_handle();
    auto inds_2 = db_indices.data_handle() + size_t(size_1);

    auto ipams              = ps.index_params;
    ipams.add_data_on_build = false;

    auto database_view =
      raft::make_device_matrix_view<DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    auto idx = ivf_pq::build<DataT, IdxT>(handle_, ipams, database_view);

    auto vecs_2_view = raft::make_device_matrix_view<DataT, IdxT>(vecs_2, size_2, ps.dim);
    auto inds_2_view = raft::make_device_vector_view<IdxT, IdxT>(inds_2, size_2);
    ivf_pq::extend<DataT, IdxT>(handle_, vecs_2_view, inds_2_view, &idx);

    auto vecs_1_view =
      raft::make_device_matrix_view<DataT, IdxT, row_major>(vecs_1, size_1, ps.dim);
    auto inds_1_view = raft::make_device_vector_view<const IdxT, IdxT>(inds_1, size_1);
    ivf_pq::extend<DataT, IdxT>(handle_, vecs_1_view, inds_1_view, &idx);
    return idx;
  }

  auto build_serialize()
  {
    ivf_pq::serialize<IdxT>(handle_, "ivf_pq_index", build_only());
    return ivf_pq::deserialize<IdxT>(handle_, "ivf_pq_index");
  }

  void check_reconstruction(const index<IdxT>& index,
                            double compression_ratio,
                            uint32_t label,
                            uint32_t n_take,
                            uint32_t n_skip)
  {
    auto& rec_list = index.lists()[label];
    auto dim       = index.dim();
    n_take         = std::min<uint32_t>(n_take, rec_list->size.load());
    n_skip         = std::min<uint32_t>(n_skip, rec_list->size.load() - n_take);

    if (n_take == 0) { return; }

    auto rec_data  = make_device_matrix<DataT>(handle_, n_take, dim);
    auto orig_data = make_device_matrix<DataT>(handle_, n_take, dim);

    ivf_pq::helpers::reconstruct_list_data(handle_, index, rec_data.view(), label, n_skip);

    matrix::gather(database.data(),
                   IdxT{dim},
                   IdxT{n_take},
                   rec_list->indices.data_handle() + n_skip,
                   IdxT{n_take},
                   orig_data.data_handle(),
                   stream_);

    compare_vectors_l2(handle_, rec_data.view(), orig_data.view(), label, compression_ratio, 0.06);
  }

  void check_reconstruct_extend(index<IdxT>* index, double compression_ratio, uint32_t label)
  {
    // NB: this is not reference, the list is retained; the index will have to create a new list on
    // `erase_list` op.
    auto old_list = index->lists()[label];
    auto n_rows   = old_list->size.load();
    if (n_rows == 0) { return; }

    auto vectors_1 = make_device_matrix<EvalT>(handle_, n_rows, index->dim());
    auto indices   = make_device_vector<IdxT>(handle_, n_rows);
    copy(indices.data_handle(), old_list->indices.data_handle(), n_rows, stream_);

    ivf_pq::helpers::reconstruct_list_data(handle_, *index, vectors_1.view(), label, 0);
    ivf_pq::helpers::erase_list(handle_, index, label);
    // NB: passing the type parameter because const->non-const implicit conversion of the mdspans
    // breaks type inference
    ivf_pq::helpers::extend_list<EvalT, IdxT>(
      handle_, index, vectors_1.view(), indices.view(), label);

    auto& new_list = index->lists()[label];
    ASSERT_NE(old_list.get(), new_list.get())
      << "The old list should have been shared and retained after ivf_pq index has erased the "
         "corresponding cluster.";

    auto vectors_2 = make_device_matrix<EvalT>(handle_, n_rows, index->dim());
    ivf_pq::helpers::reconstruct_list_data(handle_, *index, vectors_2.view(), label, 0);
    // The code search is unstable, and there's high chance of repeating values of the lvl-2 codes.
    // Hence, encoding-decoding chain often leads to altering both the PQ codes and the
    // reconstructed data.
    compare_vectors_l2(
      handle_, vectors_1.view(), vectors_2.view(), label, compression_ratio, 0.04);  // 0.025);
  }

  void check_packing(index<IdxT>* index, uint32_t label)
  {
    auto old_list = index->lists()[label];
    auto n_rows   = old_list->size.load();

    if (n_rows == 0) { return; }

    auto codes   = make_device_matrix<uint8_t>(handle_, n_rows, index->pq_dim());
    auto indices = make_device_vector<IdxT>(handle_, n_rows);
    copy(indices.data_handle(), old_list->indices.data_handle(), n_rows, stream_);

    ivf_pq::helpers::unpack_list_data(handle_, *index, codes.view(), label, 0);
    ivf_pq::helpers::erase_list(handle_, index, label);
    ivf_pq::helpers::extend_list_with_codes<IdxT>(
      handle_, index, codes.view(), indices.view(), label);

    auto& new_list = index->lists()[label];
    ASSERT_NE(old_list.get(), new_list.get())
      << "The old list should have been shared and retained after ivf_pq index has erased the "
         "corresponding cluster.";
    auto list_data_size = (n_rows / ivf_pq::kIndexGroupSize) * new_list->data.extent(1) *
                          new_list->data.extent(2) * new_list->data.extent(3);

    ASSERT_TRUE(old_list->data.size() >= list_data_size);
    ASSERT_TRUE(new_list->data.size() >= list_data_size);
    ASSERT_TRUE(devArrMatch(old_list->data.data_handle(),
                            new_list->data.data_handle(),
                            list_data_size,
                            Compare<uint8_t>{}));

    // Pack a few vectors back to the list.
    int row_offset = 9;
    int n_vec      = 3;
    ASSERT_TRUE(row_offset + n_vec < n_rows);
    size_t offset      = row_offset * index->pq_dim();
    auto codes_to_pack = make_device_matrix_view<const uint8_t, uint32_t>(
      codes.data_handle() + offset, n_vec, index->pq_dim());
    ivf_pq::helpers::pack_list_data(handle_, index, codes_to_pack, label, row_offset);
    ASSERT_TRUE(devArrMatch(old_list->data.data_handle(),
                            new_list->data.data_handle(),
                            list_data_size,
                            Compare<uint8_t>{}));

    // Another test with the API that take list_data directly
    auto list_data  = index->lists()[label]->data.view();
    uint32_t n_take = 4;
    ASSERT_TRUE(row_offset + n_take < n_rows);
    auto codes2 = raft::make_device_matrix<uint8_t>(handle_, n_take, index->pq_dim());
    ivf_pq::helpers::codepacker::unpack(
      handle_, list_data, index->pq_bits(), row_offset, codes2.view());

    // Write it back
    ivf_pq::helpers::codepacker::pack(
      handle_, make_const_mdspan(codes2.view()), index->pq_bits(), row_offset, list_data);
    ASSERT_TRUE(devArrMatch(old_list->data.data_handle(),
                            new_list->data.data_handle(),
                            list_data_size,
                            Compare<uint8_t>{}));
  }
  void check_packing_contiguous(index<IdxT>* index, uint32_t label)
  {
    auto old_list = index->lists()[label];
    auto n_rows   = old_list->size.load();

    if (n_rows == 0) { return; }

    auto codes   = make_device_matrix<uint8_t>(handle_, n_rows, index->pq_dim());
    auto indices = make_device_vector<IdxT>(handle_, n_rows);
    copy(indices.data_handle(), old_list->indices.data_handle(), n_rows, stream_);

    uint32_t code_size = ceildiv<uint32_t>(index->pq_dim() * index->pq_bits(), 8);

    auto codes_compressed = make_device_matrix<uint8_t>(handle_, n_rows, code_size);

    ivf_pq::helpers::unpack_contiguous_list_data(
      handle_, *index, codes_compressed.data_handle(), n_rows, label, 0);
    ivf_pq::helpers::erase_list(handle_, index, label);
    ivf_pq::detail::extend_list_prepare(handle_, index, make_const_mdspan(indices.view()), label);
    ivf_pq::helpers::pack_contiguous_list_data<IdxT>(
      handle_, index, codes_compressed.data_handle(), n_rows, label, 0);
    ivf_pq::helpers::recompute_internal_state(handle_, index);

    auto& new_list = index->lists()[label];
    ASSERT_NE(old_list.get(), new_list.get())
      << "The old list should have been shared and retained after ivf_pq index has erased the "
         "corresponding cluster.";
    auto list_data_size = (n_rows / ivf_pq::kIndexGroupSize) * new_list->data.extent(1) *
                          new_list->data.extent(2) * new_list->data.extent(3);

    ASSERT_TRUE(old_list->data.size() >= list_data_size);
    ASSERT_TRUE(new_list->data.size() >= list_data_size);
    ASSERT_TRUE(devArrMatch(old_list->data.data_handle(),
                            new_list->data.data_handle(),
                            list_data_size,
                            Compare<uint8_t>{}));

    // Pack a few vectors back to the list.
    uint32_t row_offset = 9;
    uint32_t n_vec      = 3;
    ASSERT_TRUE(row_offset + n_vec < n_rows);
    size_t offset      = row_offset * code_size;
    auto codes_to_pack = make_device_matrix_view<uint8_t, uint32_t>(
      codes_compressed.data_handle() + offset, n_vec, index->pq_dim());
    ivf_pq::helpers::pack_contiguous_list_data(
      handle_, index, codes_to_pack.data_handle(), n_vec, label, row_offset);
    ASSERT_TRUE(devArrMatch(old_list->data.data_handle(),
                            new_list->data.data_handle(),
                            list_data_size,
                            Compare<uint8_t>{}));

    // // Another test with the API that take list_data directly
    auto list_data  = index->lists()[label]->data.view();
    uint32_t n_take = 4;
    ASSERT_TRUE(row_offset + n_take < n_rows);
    auto codes2 = raft::make_device_matrix<uint8_t>(handle_, n_take, code_size);
    ivf_pq::helpers::codepacker::unpack_contiguous(handle_,
                                                   list_data,
                                                   index->pq_bits(),
                                                   row_offset,
                                                   n_take,
                                                   index->pq_dim(),
                                                   codes2.data_handle());

    // Write it back
    ivf_pq::helpers::codepacker::pack_contiguous(handle_,
                                                 codes2.data_handle(),
                                                 n_vec,
                                                 index->pq_dim(),
                                                 index->pq_bits(),
                                                 row_offset,
                                                 list_data);
    ASSERT_TRUE(devArrMatch(old_list->data.data_handle(),
                            new_list->data.data_handle(),
                            list_data_size,
                            Compare<uint8_t>{}));
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<IdxT> index = build_index();

    double compression_ratio =
      static_cast<double>(ps.dim * 8) / static_cast<double>(index.pq_dim() * index.pq_bits());

    for (uint32_t label = 0; label < index.n_lists(); label++) {
      switch (label % 3) {
        case 0: {
          // Reconstruct and re-write vectors for one label
          check_reconstruct_extend(&index, compression_ratio, label);
        } break;
        case 1: {
          // Dump and re-write codes for one label
          check_packing(&index, label);
          check_packing_contiguous(&index, label);
        } break;
        default: {
          // check a small subset of data in a randomly chosen cluster to see if the data
          // reconstruction works well.
          check_reconstruction(index, compression_ratio, label, 100, 7);
        }
      }
    }

    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_pq(queries_size);
    std::vector<EvalT> distances_ivf_pq(queries_size);

    rmm::device_uvector<EvalT> distances_ivf_pq_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_ivf_pq_dev(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, uint32_t>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_view = raft::make_device_matrix_view<IdxT, uint32_t>(
      indices_ivf_pq_dev.data(), ps.num_queries, ps.k);
    auto dists_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_ivf_pq_dev.data(), ps.num_queries, ps.k);

    ivf_pq::search<DataT, IdxT>(
      handle_, ps.search_params, index, query_view, inds_view, dists_view);

    update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
    update_host(indices_ivf_pq.data(), indices_ivf_pq_dev.data(), queries_size, stream_);
    resource::sync_stream(handle_);

    // A very conservative lower bound on recall
    double min_recall =
      static_cast<double>(ps.search_params.n_probes) / static_cast<double>(ps.index_params.n_lists);
    // Using a heuristic to lower the required recall due to code-packing errors
    min_recall =
      std::min(std::erfc(0.05 * compression_ratio / std::max(min_recall, 0.5)), min_recall);
    // Use explicit per-test min recall value if provided.
    min_recall = ps.min_recall.value_or(min_recall);

    ASSERT_TRUE(eval_neighbours(indices_ref,
                                indices_ivf_pq,
                                distances_ref,
                                distances_ivf_pq,
                                ps.num_queries,
                                ps.k,
                                0.0001 * compression_ratio,
                                min_recall))
      << ps;

    // Test a few extra invariants
    IdxT min_results = min_output_size(handle_, index, ps.search_params.n_probes);
    IdxT max_oob     = ps.k <= min_results ? 0 : ps.k - min_results;
    IdxT found_oob   = 0;
    for (uint32_t query_ix = 0; query_ix < ps.num_queries; query_ix++) {
      for (uint32_t k = 0; k < ps.k; k++) {
        auto flat_i   = query_ix * ps.k + k;
        auto found_ix = indices_ivf_pq[flat_i];
        if (found_ix == ivf_pq::kOutOfBoundsRecord<IdxT>) {
          found_oob++;
          continue;
        }
        ASSERT_NE(found_ix, ivf::kInvalidRecord<IdxT>)
          << "got an invalid record at query_ix = " << query_ix << ", k = " << k
          << " (distance = " << distances_ivf_pq[flat_i] << ")";
        ASSERT_LT(found_ix, ps.num_db_vecs)
          << "got an impossible index = " << found_ix << " at query_ix = " << query_ix
          << ", k = " << k << " (distance = " << distances_ivf_pq[flat_i] << ")";
      }
    }
    ASSERT_LE(found_oob, max_oob)
      << "got too many records out-of-bounds (see ivf_pq::kOutOfBoundsRecord<IdxT>).";
    if (found_oob > 0) {
      RAFT_LOG_WARN(
        "Got %zu results out-of-bounds because of large top-k (%zu) and small n_probes (%u) and "
        "small DB size/n_lists ratio (%zu / %u)",
        size_t(found_oob),
        size_t(ps.k),
        ps.search_params.n_probes,
        size_t(ps.num_db_vecs),
        ps.index_params.n_lists);
    }
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
    calc_ref();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  ivf_pq_inputs ps;                           // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
  std::vector<IdxT> indices_ref;              // NOLINT
  std::vector<EvalT> distances_ref;           // NOLINT
};

template <typename EvalT, typename DataT, typename IdxT>
class ivf_pq_filter_test : public ::testing::TestWithParam<ivf_pq_inputs> {
 public:
  ivf_pq_filter_test()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<ivf_pq_inputs>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

  void gen_data()
  {
    database.resize(size_t{ps.num_db_vecs} * size_t{ps.dim}, stream_);
    search_queries.resize(size_t{ps.num_queries} * size_t{ps.dim}, stream_);

    raft::random::RngState r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      raft::random::uniform(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0));
      raft::random::uniform(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0));
    } else {
      raft::random::uniformInt(
        handle_, r, database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20));
      raft::random::uniformInt(
        handle_, r, search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20));
    }
    resource::sync_stream(handle_);
  }

  void calc_ref()
  {
    size_t queries_size = size_t{ps.num_queries} * size_t{ps.k};
    rmm::device_uvector<EvalT> distances_naive_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
    naive_knn<EvalT, DataT, IdxT>(handle_,
                                  distances_naive_dev.data(),
                                  indices_naive_dev.data(),
                                  search_queries.data(),
                                  database.data() + test_ivf_sample_filter::offset * ps.dim,
                                  ps.num_queries,
                                  ps.num_db_vecs - test_ivf_sample_filter::offset,
                                  ps.dim,
                                  ps.k,
                                  ps.index_params.metric);
    raft::linalg::addScalar(indices_naive_dev.data(),
                            indices_naive_dev.data(),
                            IdxT(test_ivf_sample_filter::offset),
                            queries_size,
                            stream_);
    distances_ref.resize(queries_size);
    update_host(distances_ref.data(), distances_naive_dev.data(), queries_size, stream_);
    indices_ref.resize(queries_size);
    update_host(indices_ref.data(), indices_naive_dev.data(), queries_size, stream_);
    resource::sync_stream(handle_);
  }

  auto build_only()
  {
    auto ipams              = ps.index_params;
    ipams.add_data_on_build = true;

    auto index_view =
      raft::make_device_matrix_view<DataT, IdxT>(database.data(), ps.num_db_vecs, ps.dim);
    return ivf_pq::build<DataT, IdxT>(handle_, ipams, index_view);
  }

  template <typename BuildIndex>
  void run(BuildIndex build_index)
  {
    index<IdxT> index = build_index();

    double compression_ratio =
      static_cast<double>(ps.dim * 8) / static_cast<double>(index.pq_dim() * index.pq_bits());
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivf_pq(queries_size);
    std::vector<EvalT> distances_ivf_pq(queries_size);

    rmm::device_uvector<EvalT> distances_ivf_pq_dev(queries_size, stream_);
    rmm::device_uvector<IdxT> indices_ivf_pq_dev(queries_size, stream_);

    auto query_view =
      raft::make_device_matrix_view<DataT, uint32_t>(search_queries.data(), ps.num_queries, ps.dim);
    auto inds_view = raft::make_device_matrix_view<IdxT, uint32_t>(
      indices_ivf_pq_dev.data(), ps.num_queries, ps.k);
    auto dists_view = raft::make_device_matrix_view<EvalT, uint32_t>(
      distances_ivf_pq_dev.data(), ps.num_queries, ps.k);

    // Create Bitset filter
    auto removed_indices =
      raft::make_device_vector<IdxT, int64_t>(handle_, test_ivf_sample_filter::offset);
    thrust::sequence(
      resource::get_thrust_policy(handle_),
      thrust::device_pointer_cast(removed_indices.data_handle()),
      thrust::device_pointer_cast(removed_indices.data_handle() + test_ivf_sample_filter::offset));
    resource::sync_stream(handle_);

    raft::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
      handle_, removed_indices.view(), ps.num_db_vecs);
    ivf_pq::search_with_filtering<DataT, IdxT>(
      handle_,
      ps.search_params,
      index,
      query_view,
      inds_view,
      dists_view,
      raft::neighbors::filtering::bitset_filter(removed_indices_bitset.view()));

    update_host(distances_ivf_pq.data(), distances_ivf_pq_dev.data(), queries_size, stream_);
    update_host(indices_ivf_pq.data(), indices_ivf_pq_dev.data(), queries_size, stream_);
    resource::sync_stream(handle_);

    // A very conservative lower bound on recall
    double min_recall =
      static_cast<double>(ps.search_params.n_probes) / static_cast<double>(ps.index_params.n_lists);
    // Using a heuristic to lower the required recall due to code-packing errors
    min_recall =
      std::min(std::erfc(0.05 * compression_ratio / std::max(min_recall, 0.5)), min_recall);
    // Use explicit per-test min recall value if provided.
    min_recall = ps.min_recall.value_or(min_recall);

    ASSERT_TRUE(eval_neighbours(indices_ref,
                                indices_ivf_pq,
                                distances_ref,
                                distances_ivf_pq,
                                ps.num_queries,
                                ps.k,
                                0.0001 * compression_ratio,
                                min_recall))
      << ps;
  }

  void SetUp() override  // NOLINT
  {
    gen_data();
    calc_ref();
  }

  void TearDown() override  // NOLINT
  {
    cudaGetLastError();
    resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  ivf_pq_inputs ps;                           // NOLINT
  rmm::device_uvector<DataT> database;        // NOLINT
  rmm::device_uvector<DataT> search_queries;  // NOLINT
  std::vector<IdxT> indices_ref;              // NOLINT
  std::vector<EvalT> distances_ref;           // NOLINT
};

/* Test cases */
using test_cases_t = std::vector<ivf_pq_inputs>;

// concatenate parameter sets for different type
template <typename T>
auto operator+(const std::vector<T>& a, const std::vector<T>& b) -> std::vector<T>
{
  std::vector<T> res = a;
  res.insert(res.end(), b.begin(), b.end());
  return res;
}

inline auto defaults() -> test_cases_t { return {ivf_pq_inputs{}}; }

template <typename B, typename A, typename F>
auto map(const std::vector<A>& xs, F f) -> std::vector<B>
{
  std::vector<B> ys(xs.size());
  std::transform(xs.begin(), xs.end(), ys.begin(), f);
  return ys;
}

inline auto with_dims(const std::vector<uint32_t>& dims) -> test_cases_t
{
  return map<ivf_pq_inputs>(dims, [](uint32_t d) {
    ivf_pq_inputs x;
    x.dim = d;
    return x;
  });
}

/** These will surely trigger the fastest kernel available. */
inline auto small_dims() -> test_cases_t { return with_dims({1, 2, 3, 4, 5, 8, 15, 16, 17}); }

inline auto small_dims_per_cluster() -> test_cases_t
{
  return map<ivf_pq_inputs>(small_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    return y;
  });
}

inline auto big_dims() -> test_cases_t
{
  // with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053, 6144, 8192, 12288, 16384});
  auto xs = with_dims({512, 513, 1023, 1024, 1025, 2048, 2049, 2050, 2053, 6144});
  return map<ivf_pq_inputs>(xs, [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len       = 2;
    y.index_params.pq_dim = div_rounding_up_safe(x.dim, pq_len);
    // This comes from pure experimentation, also the recall depens a lot on pq_len.
    y.min_recall = 0.48 + 0.028 * std::log2(x.dim);
    return y;
  });
}

/** These will surely trigger no-smem-lut kernel.  */
inline auto big_dims_moderate_lut() -> test_cases_t
{
  return map<ivf_pq_inputs>(big_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len           = 2;
    y.index_params.pq_dim     = round_up_safe(div_rounding_up_safe(x.dim, pq_len), 4u);
    y.index_params.pq_bits    = 6;
    y.search_params.lut_dtype = CUDA_R_16F;
    y.min_recall              = 0.69;
    return y;
  });
}

/** Some of these should trigger no-basediff kernel.  */
inline auto big_dims_small_lut() -> test_cases_t
{
  return map<ivf_pq_inputs>(big_dims(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    uint32_t pq_len           = 8;
    y.index_params.pq_dim     = round_up_safe(div_rounding_up_safe(x.dim, pq_len), 4u);
    y.index_params.pq_bits    = 6;
    y.search_params.lut_dtype = CUDA_R_8U;
    y.min_recall              = 0.21;
    return y;
  });
}

/**
 * A minimal set of tests to check various enum-like parameters.
 */
inline auto enum_variety() -> test_cases_t
{
  test_cases_t xs;
#define ADD_CASE(f)                               \
  do {                                            \
    xs.push_back({});                             \
    ([](ivf_pq_inputs & x) f)(xs[xs.size() - 1]); \
  } while (0);

  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.min_recall                 = 0.86;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.min_recall                 = 0.86;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 4;
    x.min_recall                 = 0.79;
  });
  ADD_CASE({
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_bits       = 5;
    x.min_recall                 = 0.83;
  });

  ADD_CASE({
    x.index_params.pq_bits = 6;
    x.min_recall           = 0.84;
  });
  ADD_CASE({
    x.index_params.pq_bits = 7;
    x.min_recall           = 0.85;
  });
  ADD_CASE({
    x.index_params.pq_bits = 8;
    x.min_recall           = 0.86;
  });

  ADD_CASE({
    x.index_params.force_random_rotation = true;
    x.min_recall                         = 0.86;
  });
  ADD_CASE({
    x.index_params.force_random_rotation = false;
    x.min_recall                         = 0.86;
  });

  ADD_CASE({
    x.search_params.lut_dtype = CUDA_R_32F;
    x.min_recall              = 0.86;
  });
  ADD_CASE({
    x.search_params.lut_dtype = CUDA_R_16F;
    x.min_recall              = 0.86;
  });
  ADD_CASE({
    x.search_params.lut_dtype = CUDA_R_8U;
    x.min_recall              = 0.84;
  });

  ADD_CASE({
    x.search_params.internal_distance_dtype = CUDA_R_32F;
    x.min_recall                            = 0.86;
  });
  ADD_CASE({
    x.search_params.internal_distance_dtype = CUDA_R_16F;
    x.search_params.lut_dtype               = CUDA_R_16F;
    x.min_recall                            = 0.86;
  });

  return xs;
}

inline auto enum_variety_l2() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2Expanded;
    return y;
  });
}

inline auto enum_variety_ip() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    if (y.min_recall.has_value()) {
      if (y.search_params.lut_dtype == CUDA_R_8U) {
        // InnerProduct score is signed,
        // thus we're forced to used signed 8-bit representation,
        // thus we have one bit less precision
        y.min_recall = y.min_recall.value() * 0.90;
      } else {
        // In other cases it seems to perform a little bit better, still worse than L2
        y.min_recall = y.min_recall.value() * 0.94;
      }
    }
    y.index_params.metric = distance::DistanceType::InnerProduct;
    return y;
  });
}

inline auto enum_variety_l2sqrt() -> test_cases_t
{
  return map<ivf_pq_inputs>(enum_variety(), [](const ivf_pq_inputs& x) {
    ivf_pq_inputs y(x);
    y.index_params.metric = distance::DistanceType::L2SqrtExpanded;
    return y;
  });
}

/**
 * Try different number of n_probes, some of which may trigger the non-fused version of the search
 * kernel.
 */
inline auto var_n_probes() -> test_cases_t
{
  ivf_pq_inputs dflt;
  std::vector<uint32_t> xs;
  for (auto x = dflt.index_params.n_lists; x >= 1; x /= 2) {
    xs.push_back(x);
  }
  return map<ivf_pq_inputs>(xs, [](uint32_t n_probes) {
    ivf_pq_inputs x;
    x.search_params.n_probes = n_probes;
    return x;
  });
}

/**
 * Try different number of nearest neighbours.
 * Values smaller than 32 test if the code behaves well when Capacity (== 32) does not change,
 * but `k <= Capacity` changes.
 *
 * Values between `32 and ivf_pq::detail::kMaxCapacity` test various instantiations of the
 * main kernel (Capacity-templated)
 *
 * Values above ivf_pq::detail::kMaxCapacity should trigger the non-fused version of the kernel
 * (manage_local_topk = false).
 *
 * Also we test here various values that are close-but-not-power-of-two to catch any problems
 * related to rounding/alignment.
 *
 * Note, we cannot control explicitly which instance of the search kernel to choose, hence it's
 * important to try a variety of different values of `k` to make sure all paths are triggered.
 *
 * Set the log level to DEBUG (5) or above to inspect the selected kernel instances.
 */
inline auto var_k() -> test_cases_t
{
  return map<ivf_pq_inputs, uint32_t>(
    {1, 2, 3, 5, 8, 15, 16, 32, 63, 65, 127, 128, 256, 257, 1023, 2048, 2049}, [](uint32_t k) {
      ivf_pq_inputs x;
      x.k = k;
      // when there's not enough data, try more cluster probes
      x.search_params.n_probes = max(x.search_params.n_probes, min(x.index_params.n_lists, k));
      return x;
    });
}

/**
 * Cases brought up from downstream projects.
 */
inline auto special_cases() -> test_cases_t
{
  test_cases_t xs;

#define ADD_CASE(f)                               \
  do {                                            \
    xs.push_back({});                             \
    ([](ivf_pq_inputs & x) f)(xs[xs.size() - 1]); \
  } while (0);

  ADD_CASE({
    x.num_db_vecs                = 1183514;
    x.dim                        = 100;
    x.num_queries                = 10000;
    x.k                          = 10;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_dim        = 10;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 1024;
    x.search_params.n_probes     = 50;
  });

  ADD_CASE({
    x.num_db_vecs                = 10000;
    x.dim                        = 16;
    x.num_queries                = 500;
    x.k                          = 128;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 100;
    x.search_params.n_probes     = 100;
  });

  ADD_CASE({
    x.num_db_vecs                = 10000;
    x.dim                        = 16;
    x.num_queries                = 500;
    x.k                          = 129;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 100;
    x.search_params.n_probes     = 100;
  });

  ADD_CASE({
    x.num_db_vecs                = 4335;
    x.dim                        = 4;
    x.num_queries                = 100000;
    x.k                          = 12;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_SUBSPACE;
    x.index_params.pq_dim        = 2;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 69;
    x.search_params.n_probes     = 69;
  });

  ADD_CASE({
    x.num_db_vecs                = 4335;
    x.dim                        = 4;
    x.num_queries                = 100000;
    x.k                          = 12;
    x.index_params.metric        = distance::DistanceType::L2Expanded;
    x.index_params.codebook_kind = ivf_pq::codebook_gen::PER_CLUSTER;
    x.index_params.pq_dim        = 2;
    x.index_params.pq_bits       = 8;
    x.index_params.n_lists       = 69;
    x.search_params.n_probes     = 69;
  });

  return xs;
}

/* Test instantiations */

#define TEST_BUILD_SEARCH(type)                         \
  TEST_P(type, build_search) /* NOLINT */               \
  {                                                     \
    this->run([this]() { return this->build_only(); }); \
  }

#define TEST_BUILD_EXTEND_SEARCH(type)                       \
  TEST_P(type, build_extend_search) /* NOLINT */             \
  {                                                          \
    this->run([this]() { return this->build_2_extends(); }); \
  }

#define TEST_BUILD_SERIALIZE_SEARCH(type)                    \
  TEST_P(type, build_serialize_search) /* NOLINT */          \
  {                                                          \
    this->run([this]() { return this->build_serialize(); }); \
  }

#define INSTANTIATE(type, vals) \
  INSTANTIATE_TEST_SUITE_P(IvfPq, type, ::testing::ValuesIn(vals)); /* NOLINT */

}  // namespace raft::neighbors::ivf_pq
