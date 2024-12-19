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
  if (params.sparsity == 0.0) { os << "<-inner product for comparison"; }
  return os;
}

template <typename value_t,
          bool bitmap_or_bitset = true,
          typename index_t      = int64_t,
          typename bits_t       = uint32_t>
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
      bits_d(0, stream),
      c_dense_data_d(0, stream)
  {
    index_t element = raft::ceildiv(index_t(params.m * params.n), index_t(sizeof(bits_t) * 8));
    std::vector<bits_t> bits_h(element);

    a_data_d.resize(params.m * params.k, stream);
    b_data_d.resize(params.k * params.n, stream);
    bits_d.resize(element, stream);

    raft::random::RngState rng(2024ULL);
    raft::random::uniform(
      handle, rng, a_data_d.data(), params.m * params.k, value_t(-1.0), value_t(1.0));
    raft::random::uniform(
      handle, rng, b_data_d.data(), params.k * params.n, value_t(-1.0), value_t(1.0));

    std::vector<bool> c_dense_data_h(params.m * params.n);

    if constexpr (bitmap_or_bitset) {
      c_true_nnz = create_sparse_matrix(params.m, params.n, params.sparsity, bits_h);
    } else {
      c_true_nnz = create_sparse_matrix(1, params.n, params.sparsity, bits_h);
      repeat_cpu_bitset_inplace(bits_h, params.n, params.m - 1);
      c_true_nnz *= params.m;
    }

    std::vector<value_t> values(c_true_nnz);
    std::vector<index_t> indices(c_true_nnz);
    std::vector<index_t> indptr(params.m + 1);

    c_data_d.resize(c_true_nnz, stream);
    c_indptr_d.resize(params.m + 1, stream);
    c_indices_d.resize(c_true_nnz, stream);
    c_dense_data_d.resize(params.m * params.n, stream);

    cpu_convert_to_csr(bits_h, params.m, params.n, indices, indptr);
    RAFT_EXPECTS(c_true_nnz == c_indices_d.size(),
                 "Something wrong. The c_true_nnz != c_indices_d.size()!");

    update_device(c_data_d.data(), values.data(), c_true_nnz, stream);
    update_device(c_indices_d.data(), indices.data(), c_true_nnz, stream);
    update_device(c_indptr_d.data(), indptr.data(), params.m + 1, stream);
    update_device(bits_d.data(), bits_h.data(), element, stream);
  }

  void repeat_cpu_bitset_inplace(std::vector<bits_t>& inout, size_t input_bits, size_t repeat)
  {
    size_t output_bit_index = input_bits;

    for (size_t r = 0; r < repeat; ++r) {
      for (size_t i = 0; i < input_bits; ++i) {
        size_t input_unit_index = i / (sizeof(bits_t) * 8);
        size_t input_bit_offset = i % (sizeof(bits_t) * 8);
        bool bit                = (inout[input_unit_index] >> input_bit_offset) & 1;

        size_t output_unit_index = output_bit_index / (sizeof(bits_t) * 8);
        size_t output_bit_offset = output_bit_index % (sizeof(bits_t) * 8);

        inout[output_unit_index] |= (static_cast<bits_t>(bit) << output_bit_offset);

        ++output_bit_index;
      }
    }
  }

  index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bits_t>& bits)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * (1.0f - sparsity));
    index_t res      = num_ones;

    if (sparsity == 0.0f) {
      std::fill(bits.begin(), bits.end(), 0xffffffff);
      return num_ones;
    }

    for (auto& item : bits) {
      item = static_cast<bits_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bits_t& element      = bits[index / (8 * sizeof(bits_t))];
      index_t bit_position = index % (8 * sizeof(bits_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void cpu_convert_to_csr(std::vector<bits_t>& bits,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bits_t element       = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bits[index / (8 * sizeof(bits_t))];
        bit_position = index % (8 * sizeof(bits_t));

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

    auto c = raft::make_device_csr_matrix_view<value_t>(c_data_d.data(), c_structure);

    if (params.sparsity > 0.0) {
      if constexpr (bitmap_or_bitset) {
        auto mask =
          raft::core::bitmap_view<const bits_t, index_t>(bits_d.data(), params.m, params.n);
        raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
      } else {
        auto mask = raft::core::bitset_view<const bits_t, index_t>(bits_d.data(), params.n);
        raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
      }
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

    loop_on_state(state, [this, &a, &b, &c]() {
      if (params.sparsity > 0.0) {
        if constexpr (bitmap_or_bitset) {
          auto mask =
            raft::core::bitmap_view<const bits_t, index_t>(bits_d.data(), params.m, params.n);
          raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
        } else {
          auto mask = raft::core::bitset_view<const bits_t, index_t>(bits_d.data(), params.n);
          raft::sparse::linalg::masked_matmul(handle, a, b, mask, c);
        }
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
  rmm::device_uvector<bits_t> bits_d;

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
                                               {0.99f, 0.9f, 0.8f, 0.5f, 0.0f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(
      MaskedMatmulBenchParams<value_t>({params.m, params.k, params.n, params.sparsity}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((MaskedMatmulBench<float, true>), "", getInputs<float>());
RAFT_BENCH_REGISTER((MaskedMatmulBench<float, false>), "", getInputs<float>());

}  // namespace raft::bench::linalg
