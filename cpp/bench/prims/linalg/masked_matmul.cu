/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <common/benchmark.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/linalg/masked_matmul.hpp>
#include <raft/util/itertools.hpp>

#include <cusparse_v2.h>

#include <random>
#include <sstream>
#include <vector>

namespace raft::bench::linalg {

template <typename value_t>
struct MaskedMatmulBenchParams {
  size_t m;
  size_t k;
  size_t n;
  float sparsity;
  value_t alpha = 1.0;
  value_t beta  = 0.0;
};

template <typename value_t>
inline auto operator<<(std::ostream& os, const MaskedMatmulBenchParams<value_t>& params)
  -> std::ostream&
{
  os << " m*k*n=" << params.m << "*" << params.k << "*" << params.n
     << "\tsparsity=" << params.sparsity;
  if (params.sparsity == 1.0) { os << "<-inner product for comparison"; }
  return os;
}

template <typename value_t, typename index_t = int64_t, typename bitmap_t = uint32_t>
struct MaskedMatmulBench : public fixture {
  MaskedMatmulBench(const MaskedMatmulBenchParams<value_t>& p)
    : fixture(true),
      params(p),
      handle(stream),
      a_data_d(0, stream),
      b_data_d(0, stream),
      c_indptr_d(0, stream),
      c_indices_d(0, stream),
      c_data_d(0, stream),
      bitmap_d(0, stream),
      c_dense_data_d(0, stream)
  {
    index_t element = raft::ceildiv(index_t(params.m * params.n), index_t(sizeof(bitmap_t) * 8));
    std::vector<bitmap_t> bitmap_h(element);

    a_data_d.resize(params.m * params.k, stream);
    b_data_d.resize(params.k * params.n, stream);
    bitmap_d.resize(element, stream);

    raft::random::RngState rng(2024ULL);
    raft::random::uniform(
      handle, rng, a_data_d.data(), params.m * params.k, value_t(-1.0), value_t(1.0));
    raft::random::uniform(
      handle, rng, b_data_d.data(), params.k * params.n, value_t(-1.0), value_t(1.0));

    std::vector<bool> c_dense_data_h(params.m * params.n);

    c_true_nnz = create_sparse_matrix(params.m, params.n, params.sparsity, bitmap_h);

    std::vector<value_t> values(c_true_nnz);
    std::vector<index_t> indices(c_true_nnz);
    std::vector<index_t> indptr(params.m + 1);

    c_data_d.resize(c_true_nnz, stream);
    c_indptr_d.resize(params.m + 1, stream);
    c_indices_d.resize(c_true_nnz, stream);
    c_dense_data_d.resize(params.m * params.n, stream);

    cpu_convert_to_csr(bitmap_h, params.m, params.n, indices, indptr);
    RAFT_EXPECTS(c_true_nnz == c_indices_d.size(),
                 "Something wrong. The c_true_nnz != c_indices_d.size()!");

    update_device(c_data_d.data(), values.data(), c_true_nnz, stream);
    update_device(c_indices_d.data(), indices.data(), c_true_nnz, stream);
    update_device(c_indptr_d.data(), indptr.data(), params.m + 1, stream);
    update_device(bitmap_d.data(), bitmap_h.data(), element, stream);
  }

  index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bitmap_t>& bitmap)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
    index_t res      = num_ones;

    for (auto& item : bitmap) {
      item = static_cast<bitmap_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bitmap_t& element    = bitmap[index / (8 * sizeof(bitmap_t))];
      index_t bit_position = index % (8 * sizeof(bitmap_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void cpu_convert_to_csr(std::vector<bitmap_t>& bitmap,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bitmap_t element     = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bitmap[index / (8 * sizeof(bitmap_t))];
        bit_position = index % (8 * sizeof(bitmap_t));

        if (((element >> bit_position) & 1)) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  ~MaskedMatmulBench() {}

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    auto a = raft::make_device_matrix_view<const value_t, index_t, row_major>(
      a_data_d.data(), params.m, params.k);

    auto b = raft::make_device_matrix_view<const value_t, index_t, row_major>(
      b_data_d.data(), params.n, params.k);

    auto c_structure = raft::make_device_compressed_structure_view<int64_t, int64_t, int64_t>(
      c_indptr_d.data(),
      c_indices_d.data(),
      params.m,
      params.n,
      static_cast<index_t>(c_indices_d.size()));

    auto mask =
      raft::core::bitmap_view<const bitmap_t, index_t>(bitmap_d.data(), params.m, params.n);

    auto c = raft::make_device_csr_matrix_view<value_t>(c_data_d.data(), c_structure);

    if (params.sparsity < 1.0) {
      raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
    } else {
      raft::distance::pairwise_distance(handle,
                                        a_data_d.data(),
                                        b_data_d.data(),
                                        c_dense_data_d.data(),
                                        static_cast<int>(params.m),
                                        static_cast<int>(params.n),
                                        static_cast<int>(params.k),
                                        raft::distance::DistanceType::InnerProduct,
                                        true);
    }
    resource::sync_stream(handle);

    raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
    resource::sync_stream(handle);

    loop_on_state(state, [this, &a, &b, &mask, &c]() {
      if (params.sparsity < 1.0) {
        raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
      } else {
        raft::distance::pairwise_distance(handle,
                                          a_data_d.data(),
                                          b_data_d.data(),
                                          c_dense_data_d.data(),
                                          static_cast<int>(params.m),
                                          static_cast<int>(params.n),
                                          static_cast<int>(params.k),
                                          raft::distance::DistanceType::InnerProduct,
                                          true);
      }
      resource::sync_stream(handle);
    });
  }

 private:
  const raft::device_resources handle;
  MaskedMatmulBenchParams<value_t> params;

  rmm::device_uvector<value_t> a_data_d;
  rmm::device_uvector<value_t> b_data_d;
  rmm::device_uvector<bitmap_t> bitmap_d;

  rmm::device_uvector<value_t> c_dense_data_d;

  size_t c_true_nnz = 0;
  rmm::device_uvector<index_t> c_indptr_d;
  rmm::device_uvector<index_t> c_indices_d;
  rmm::device_uvector<value_t> c_data_d;
};

template <typename value_t>
static std::vector<MaskedMatmulBenchParams<value_t>> getInputs()
{
  std::vector<MaskedMatmulBenchParams<value_t>> param_vec;
  struct TestParams {
    size_t m;
    size_t k;
    size_t n;
    float sparsity;
  };

  const std::vector<TestParams> params_group =
    raft::util::itertools::product<TestParams>({size_t(10), size_t(1024)},
                                               {size_t(128), size_t(1024)},
                                               {size_t(1024 * 1024)},
                                               {0.01f, 0.1f, 0.2f, 0.5f, 1.0f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(
      MaskedMatmulBenchParams<value_t>({params.m, params.k, params.n, params.sparsity}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((MaskedMatmulBench<float>), "", getInputs<float>());

}  // namespace raft::bench::linalg
