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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/select_k.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

#include <random>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace raft::bench::sparse {

template <typename index_t>
struct bench_param {
  index_t n_rows;
  index_t n_cols;
  index_t top_k;
  float sparsity;
  bool select_min         = true;
  bool customized_indices = false;
};

template <typename index_t>
inline auto operator<<(std::ostream& os, const bench_param<index_t>& params) -> std::ostream&
{
  os << " rows*cols=" << params.n_rows << "*" << params.n_cols << "\ttop_k=" << params.top_k
     << "\tsparsity=" << params.sparsity;
  return os;
}

template <typename value_t, typename index_t>
struct SelectKCsrTest : public fixture {
  SelectKCsrTest(const bench_param<index_t>& p)
    : fixture(true),
      params(p),
      handle(stream),
      values_d(0, stream),
      indptr_d(0, stream),
      indices_d(0, stream),
      customized_indices_d(0, stream),
      dst_values_d(0, stream),
      dst_indices_d(0, stream)
  {
    std::vector<bool> dense_values_h(params.n_rows * params.n_cols, false);
    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, dense_values_h);

    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> customized_indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_rows + 1);

    convert_to_csr(dense_values_h, params.n_rows, params.n_cols, indices_h, indptr_h);

    std::vector<value_t> dst_values_h(params.n_rows * params.top_k, static_cast<value_t>(2.0f));
    std::vector<index_t> dst_indices_h(params.n_rows * params.top_k,
                                       static_cast<index_t>(params.n_rows * params.n_cols * 100));

    dst_values_d.resize(params.n_rows * params.top_k, stream);
    dst_indices_d.resize(params.n_rows * params.top_k, stream);
    values_d.resize(nnz, stream);

    if (nnz) {
      auto blobs_values = raft::make_device_matrix<value_t, index_t>(handle, 1, nnz);
      auto labels       = raft::make_device_vector<index_t, index_t>(handle, 1);

      raft::random::make_blobs<value_t, index_t>(blobs_values.data_handle(),
                                                 labels.data_handle(),
                                                 1,
                                                 nnz,
                                                 1,
                                                 stream,
                                                 false,
                                                 nullptr,
                                                 nullptr,
                                                 value_t(1.0),
                                                 false,
                                                 value_t(-10.0f),
                                                 value_t(10.0f),
                                                 uint64_t(2024));
      raft::copy(values_d.data(), blobs_values.data_handle(), nnz, stream);
      resource::sync_stream(handle);
    }

    indices_d.resize(nnz, stream);
    indptr_d.resize(params.n_rows + 1, stream);

    update_device(indices_d.data(), indices_h.data(), indices_h.size(), stream);
    update_device(indptr_d.data(), indptr_h.data(), indptr_h.size(), stream);

    if (params.customized_indices) {
      customized_indices_d.resize(nnz, stream);
      update_device(customized_indices_d.data(),
                    customized_indices_h.data(),
                    customized_indices_h.size(),
                    stream);
    }
  }

  index_t create_sparse_matrix(index_t m, index_t n, value_t sparsity, std::vector<bool>& matrix)
  {
    index_t total_elements = static_cast<index_t>(m * n);
    index_t num_ones       = static_cast<index_t>((total_elements * 1.0f) * sparsity);
    index_t res            = num_ones;

    for (index_t i = 0; i < total_elements; ++i) {
      matrix[i] = false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_idx(0, total_elements - 1);

    while (num_ones > 0) {
      size_t index = dis_idx(gen);
      if (matrix[index] == false) {
        matrix[index] = true;
        num_ones--;
      }
    }
    return res;
  }

  void convert_to_csr(std::vector<bool>& matrix,
                      index_t rows,
                      index_t cols,
                      std::vector<index_t>& indices,
                      std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        if (matrix[i * cols + j]) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  template <typename data_t>
  std::optional<data_t> get_opt_var(data_t x)
  {
    if (params.customized_indices) {
      return x;
    } else {
      return std::nullopt;
    }
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    auto in_val_structure = raft::make_device_compressed_structure_view<index_t, index_t, index_t>(
      indptr_d.data(),
      indices_d.data(),
      params.n_rows,
      params.n_cols,
      static_cast<index_t>(indices_d.size()));

    auto in_val =
      raft::make_device_csr_matrix_view<const value_t>(values_d.data(), in_val_structure);

    std::optional<raft::device_vector_view<const index_t, index_t>> in_idx;

    in_idx = get_opt_var(
      raft::make_device_vector_view<const index_t, index_t>(customized_indices_d.data(), nnz));

    auto out_val = raft::make_device_matrix_view<value_t, index_t, raft::row_major>(
      dst_values_d.data(), params.n_rows, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      dst_indices_d.data(), params.n_rows, params.top_k);

    raft::matrix::select_k(handle, in_val, in_idx, out_val, out_idx, params.select_min);
    resource::sync_stream(handle);
    loop_on_state(state, [this, &in_val, &in_idx, &out_val, &out_idx]() {
      raft::matrix::select_k(handle, in_val, in_idx, out_val, out_idx, params.select_min);
      resource::sync_stream(handle);
    });
  }

 protected:
  const raft::device_resources handle;

  bench_param<index_t> params;
  index_t nnz;

  rmm::device_uvector<value_t> values_d;
  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;
  rmm::device_uvector<index_t> customized_indices_d;

  rmm::device_uvector<value_t> dst_values_d;
  rmm::device_uvector<index_t> dst_indices_d;
};  // struct SelectKCsrTest

template <typename index_t>
const std::vector<bench_param<index_t>> getInputs()
{
  std::vector<bench_param<index_t>> param_vec;
  struct TestParams {
    index_t m;
    index_t n;
    index_t k;
    float sparsity;
  };

  const std::vector<TestParams> params_group =
    raft::util::itertools::product<TestParams>({index_t(10), index_t(1024)},
                                               {index_t(1024 * 10), index_t(1024 * 1024)},
                                               {index_t(128), index_t(100), index_t(2048)},
                                               {0.1f, 0.2f, 0.5f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.k, params.sparsity}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((SelectKCsrTest<float, int>), "", getInputs<int>());

}  // namespace raft::bench::sparse
