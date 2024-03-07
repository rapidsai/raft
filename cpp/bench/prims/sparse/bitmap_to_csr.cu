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
  index_t n_rows;
  index_t n_cols;
  float sparsity;
};

template <typename index_t>
inline auto operator<<(std::ostream& os, const bench_param<index_t>& params) -> std::ostream&
{
  os << " rows*cols=" << params.n_rows << "*" << params.n_cols << "\tsparsity=" << params.sparsity;
  return os;
}

template <typename bitmap_t, typename index_t, typename value_t = float>
struct BitmapToCsrBench : public fixture {
  BitmapToCsrBench(const bench_param<index_t>& p)
    : fixture(true),
      params(p),
      handle(stream),
      bitmap_d(0, stream),
      nnz(0),
      indptr_d(0, stream),
      indices_d(0, stream),
      values_d(0, stream)
  {
    index_t element = raft::ceildiv(params.n_rows * params.n_cols, index_t(sizeof(bitmap_t) * 8));
    std::vector<bitmap_t> bitmap_h(element);
    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, bitmap_h);

    bitmap_d.resize(bitmap_h.size(), stream);
    indptr_d.resize(params.n_rows + 1, stream);
    indices_d.resize(nnz, stream);
    values_d.resize(nnz, stream);

    update_device(bitmap_d.data(), bitmap_h.data(), bitmap_h.size(), stream);

    resource::sync_stream(handle);
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

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    auto bitmap =
      raft::core::bitmap_view<bitmap_t, index_t>(bitmap_d.data(), params.n_rows, params.n_cols);

    auto csr_view = raft::make_device_compressed_structure_view<index_t, index_t, index_t>(
      indptr_d.data(), indices_d.data(), params.n_rows, params.n_cols, nnz);
    auto csr = raft::make_device_csr_matrix_view<value_t, index_t>(values_d.data(), csr_view);

    raft::sparse::convert::bitmap_to_csr<bitmap_t, index_t>(handle, bitmap, csr);

    resource::sync_stream(handle);
    loop_on_state(state, [this, &bitmap, &csr]() {
      raft::sparse::convert::bitmap_to_csr<bitmap_t, index_t>(handle, bitmap, csr);
    });
  }

 protected:
  const raft::device_resources handle;

  bench_param<index_t> params;

  rmm::device_uvector<bitmap_t> bitmap_d;
  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;
  rmm::device_uvector<value_t> values_d;

  index_t nnz;
};  // struct BitmapToCsrBench

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
    {index_t(10), index_t(1024)}, {index_t(1024 * 1024)}, {0.01f, 0.1f, 0.2f, 0.5f});

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.sparsity}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((BitmapToCsrBench<uint32_t, int, float>), "", getInputs<int>());
RAFT_BENCH_REGISTER((BitmapToCsrBench<uint64_t, int, double>), "", getInputs<int>());

}  // namespace raft::bench::sparse
