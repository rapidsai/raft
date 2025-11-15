/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <common/benchmark.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

#include <sstream>
#include <vector>

namespace raft::bench::sparse {

template <typename index_t>
struct bench_param {
  index_t n_repeat;
  index_t n_cols;
  float sparsity;
};

template <typename index_t>
inline auto operator<<(std::ostream& os, const bench_param<index_t>& params) -> std::ostream&
{
  os << " rows*cols=" << params.n_repeat << "*" << params.n_cols
     << "\tsparsity=" << params.sparsity;
  return os;
}

template <typename bitset_t, typename index_t, typename value_t = float>
struct BitsetToCsrBench : public fixture {
  BitsetToCsrBench(const bench_param<index_t>& p)
    : fixture(true),
      params(p),
      handle(stream),
      bitset_d(0, stream),
      nnz(0),
      indptr_d(0, stream),
      indices_d(0, stream),
      values_d(0, stream)
  {
    index_t element = raft::ceildiv(1 * params.n_cols, index_t(sizeof(bitset_t) * 8));
    std::vector<bitset_t> bitset_h(element);
    nnz = create_sparse_matrix(1, params.n_cols, params.sparsity, bitset_h);

    bitset_d.resize(bitset_h.size(), stream);
    indptr_d.resize(params.n_repeat + 1, stream);
    indices_d.resize(nnz, stream);
    values_d.resize(nnz, stream);

    update_device(bitset_d.data(), bitset_h.data(), bitset_h.size(), stream);

    resource::sync_stream(handle);
  }

  index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bitset_t>& bitset)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * (1.0f - sparsity));
    index_t res      = num_ones;

    for (auto& item : bitset) {
      item = static_cast<bitset_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bitset_t& element    = bitset[index / (8 * sizeof(bitset_t))];
      index_t bit_position = index % (8 * sizeof(bitset_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    auto bitset = raft::core::bitset_view<bitset_t, index_t>(bitset_d.data(), 1 * params.n_cols);

    auto csr_view = raft::make_device_compressed_structure_view<index_t, index_t, index_t>(
      indptr_d.data(), indices_d.data(), params.n_repeat, params.n_cols, nnz);
    auto csr = raft::make_device_csr_matrix<value_t, index_t>(handle, csr_view);

    raft::sparse::convert::bitset_to_csr<bitset_t, index_t>(handle, bitset, csr);

    resource::sync_stream(handle);
    loop_on_state(state, [this, &bitset, &csr]() {
      raft::sparse::convert::bitset_to_csr<bitset_t, index_t>(handle, bitset, csr);
    });
  }

 protected:
  const raft::device_resources handle;

  bench_param<index_t> params;

  rmm::device_uvector<bitset_t> bitset_d;
  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;
  rmm::device_uvector<value_t> values_d;

  index_t nnz;
};  // struct BitsetToCsrBench

template <typename index_t>
const std::vector<bench_param<index_t>> getInputs()
{
  std::vector<bench_param<index_t>> param_vec;
  struct TestParams {
    index_t m;
    index_t n;
    float sparsity;
  };

  const std::vector<TestParams> params_group = raft::util::itertools::product<TestParams>(
    {index_t(10), index_t(1024)}, {index_t(1024 * 1024)}, {0.99f, 0.9f, 0.8f, 0.5f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.sparsity}));
  }
  return param_vec;
}

template <typename index_t = int64_t>
const std::vector<bench_param<index_t>> getLargeInputs()
{
  std::vector<bench_param<index_t>> param_vec;
  struct TestParams {
    index_t m;
    index_t n;
    float sparsity;
  };

  const std::vector<TestParams> params_group = raft::util::itertools::product<TestParams>(
    {index_t(1), index_t(100)}, {index_t(100 * 1000000)}, {0.95f, 0.99f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.sparsity}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((BitsetToCsrBench<uint32_t, int, float>), "", getInputs<int>());
RAFT_BENCH_REGISTER((BitsetToCsrBench<uint64_t, int, double>), "", getInputs<int>());

RAFT_BENCH_REGISTER((BitsetToCsrBench<uint32_t, int64_t, float>), "", getLargeInputs<int64_t>());

}  // namespace raft::bench::sparse
