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
#include <raft/sparse/linalg/sddmm.hpp>
#include <raft/util/itertools.hpp>

#include <cusparse_v2.h>

#include <random>
#include <sstream>
#include <vector>

namespace raft::bench::linalg {

template <typename ValueType>
struct SDDMMBenchParams {
  size_t m;
  size_t k;
  size_t n;
  float sparsity;
  bool transpose_a;
  bool transpose_b;
  ValueType alpha = 1.0;
  ValueType beta  = 0.0;
};

enum Alg { SDDMM, Inner };

template <typename ValueType>
inline auto operator<<(std::ostream& os, const SDDMMBenchParams<ValueType>& params) -> std::ostream&
{
  os << " m*k*n=" << params.m << "*" << params.k << "*" << params.n
     << "\tsparsity=" << params.sparsity << "\ttrans_a=" << (params.transpose_a ? "T" : "F")
     << " trans_b=" << (params.transpose_b ? "T" : "F");
  return os;
}

template <typename ValueType,
          typename LayoutPolicyA = row_major,
          typename LayoutPolicyB = col_major,
          const int SDDMMorInner = Alg::SDDMM,
          typename IndexType     = int64_t>
struct SDDMMBench : public fixture {
  SDDMMBench(const SDDMMBenchParams<ValueType>& p)
    : fixture(true),
      params(p),
      handle(stream),
      a_data_d(0, stream),
      b_data_d(0, stream),
      c_indptr_d(0, stream),
      c_indices_d(0, stream),
      c_data_d(0, stream),
      c_dense_data_d(0, stream)
  {
    a_data_d.resize(params.m * params.k, stream);
    b_data_d.resize(params.k * params.n, stream);

    raft::random::RngState rng(2024ULL);
    raft::random::uniform(
      handle, rng, a_data_d.data(), params.m * params.k, ValueType(-1.0), ValueType(1.0));
    raft::random::uniform(
      handle, rng, b_data_d.data(), params.k * params.n, ValueType(-1.0), ValueType(1.0));

    std::vector<bool> c_dense_data_h(params.m * params.n);

    c_true_nnz = create_sparse_matrix(c_dense_data_h);
    std::vector<ValueType> values(c_true_nnz);
    std::vector<IndexType> indices(c_true_nnz);
    std::vector<IndexType> indptr(params.m + 1);

    c_data_d.resize(c_true_nnz, stream);
    c_indptr_d.resize(params.m + 1, stream);
    c_indices_d.resize(c_true_nnz, stream);

    if (SDDMMorInner == Alg::Inner) { c_dense_data_d.resize(params.m * params.n, stream); }

    convert_to_csr(c_dense_data_h, params.m, params.n, values, indices, indptr);
    RAFT_EXPECTS(c_true_nnz == c_indices_d.size(),
                 "Something wrong. The c_true_nnz != c_indices_d.size()!");

    update_device(c_data_d.data(), values.data(), c_true_nnz, stream);
    update_device(c_indices_d.data(), indices.data(), c_true_nnz, stream);
    update_device(c_indptr_d.data(), indptr.data(), params.m + 1, stream);
  }

  void convert_to_csr(std::vector<bool>& matrix,
                      IndexType rows,
                      IndexType cols,
                      std::vector<ValueType>& values,
                      std::vector<IndexType>& indices,
                      std::vector<IndexType>& indptr)
  {
    IndexType offset_indptr = 0;
    IndexType offset_values = 0;
    indptr[offset_indptr++] = 0;

    for (IndexType i = 0; i < rows; ++i) {
      for (IndexType j = 0; j < cols; ++j) {
        if (matrix[i * cols + j]) {
          values[offset_values]  = static_cast<ValueType>(1.0);
          indices[offset_values] = static_cast<IndexType>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<IndexType>(offset_values);
    }
  }

  size_t create_sparse_matrix(std::vector<bool>& matrix)
  {
    size_t total_elements = static_cast<size_t>(params.m * params.n);
    size_t num_ones       = static_cast<size_t>((total_elements * 1.0f) * params.sparsity);
    size_t res            = num_ones;

    for (size_t i = 0; i < total_elements; ++i) {
      matrix[i] = false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, total_elements - 1);

    while (num_ones > 0) {
      size_t index = dis(gen);

      if (matrix[index] == false) {
        matrix[index] = true;
        num_ones--;
      }
    }
    return res;
  }

  ~SDDMMBench() {}

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    auto a = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyA>(
      a_data_d.data(),
      (!params.transpose_a ? params.m : params.k),
      (!params.transpose_a ? params.k : params.m));

    auto b = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyB>(
      b_data_d.data(),
      (!params.transpose_b ? params.k : params.n),
      (!params.transpose_b ? params.n : params.k));

    auto c_structure = raft::make_device_compressed_structure_view<int64_t, int64_t, int64_t>(
      c_indptr_d.data(),
      c_indices_d.data(),
      params.m,
      params.n,
      static_cast<IndexType>(c_indices_d.size()));

    auto c = raft::make_device_csr_matrix_view<ValueType>(c_data_d.data(), c_structure);
    raft::resource::get_cusparse_handle(handle);

    resource::sync_stream(handle);

    auto op_a = params.transpose_a ? raft::linalg::Operation::TRANSPOSE
                                   : raft::linalg::Operation::NON_TRANSPOSE;
    auto op_b = params.transpose_b ? raft::linalg::Operation::TRANSPOSE
                                   : raft::linalg::Operation::NON_TRANSPOSE;

    raft::sparse::linalg::sddmm(handle,
                                a,
                                b,
                                c,
                                op_a,
                                op_b,
                                raft::make_host_scalar_view<ValueType>(&params.alpha),
                                raft::make_host_scalar_view<ValueType>(&params.beta));
    resource::sync_stream(handle);

    loop_on_state(state, [this, &a, &b, &c, &op_a, &op_b]() {
      if (SDDMMorInner == Alg::SDDMM) {
        raft::sparse::linalg::sddmm(handle,
                                    a,
                                    b,
                                    c,
                                    op_a,
                                    op_b,
                                    raft::make_host_scalar_view<ValueType>(&params.alpha),
                                    raft::make_host_scalar_view<ValueType>(&params.beta));
        resource::sync_stream(handle);
      } else {
        raft::distance::pairwise_distance(handle,
                                          a_data_d.data(),
                                          b_data_d.data(),
                                          c_dense_data_d.data(),
                                          static_cast<int>(params.m),
                                          static_cast<int>(params.n),
                                          static_cast<int>(params.k),
                                          raft::distance::DistanceType::InnerProduct,
                                          std::is_same_v<LayoutPolicyA, row_major>);
        resource::sync_stream(handle);
      }
    });
  }

 private:
  const raft::device_resources handle;
  SDDMMBenchParams<ValueType> params;

  rmm::device_uvector<ValueType> a_data_d;
  rmm::device_uvector<ValueType> b_data_d;
  rmm::device_uvector<ValueType> c_dense_data_d;

  size_t c_true_nnz = 0;
  rmm::device_uvector<IndexType> c_indptr_d;
  rmm::device_uvector<IndexType> c_indices_d;
  rmm::device_uvector<ValueType> c_data_d;
};

template <typename ValueType>
static std::vector<SDDMMBenchParams<ValueType>> getInputs()
{
  std::vector<SDDMMBenchParams<ValueType>> param_vec;
  struct TestParams {
    bool transpose_a;
    bool transpose_b;
    size_t m;
    size_t k;
    size_t n;
    float sparsity;
  };

  const std::vector<TestParams> params_group =
    raft::util::itertools::product<TestParams>({false, true},
                                               {false, true},
                                               {size_t(10), size_t(1024)},
                                               {size_t(128), size_t(1024)},
                                               {size_t(1024 * 1024)},
                                               {0.01f, 0.1f, 0.2f, 0.5f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(SDDMMBenchParams<ValueType>(
      {params.m, params.k, params.n, params.sparsity, params.transpose_a, params.transpose_b}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((SDDMMBench<float, row_major, col_major, Alg::SDDMM>), "", getInputs<float>());
RAFT_BENCH_REGISTER((SDDMMBench<float, col_major, row_major, Alg::SDDMM>), "", getInputs<float>());
RAFT_BENCH_REGISTER((SDDMMBench<float, row_major, row_major, Alg::SDDMM>), "", getInputs<float>());
RAFT_BENCH_REGISTER((SDDMMBench<float, col_major, col_major, Alg::SDDMM>), "", getInputs<float>());

RAFT_BENCH_REGISTER((SDDMMBench<float, row_major, col_major, Alg::Inner>), "", getInputs<float>());

}  // namespace raft::bench::linalg
