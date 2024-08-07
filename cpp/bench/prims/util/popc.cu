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

#include <raft/util/popc.cuh>

namespace raft::bench::core {

template <typename index_t>
struct PopcInputs {
  index_t n_rows;
  index_t n_cols;
  float sparsity;
};

template <typename index_t>
inline auto operator<<(std::ostream& os, const PopcInputs<index_t>& params) -> std::ostream&
{
  os << params.n_rows << "#" << params.n_cols << "#" << params.sparsity;
  return os;
}

template <typename index_t, typename bits_t = uint32_t>
struct popc_bench : public fixture {
  popc_bench(const PopcInputs<index_t>& p)
    : params(p),
      n_element(raft::ceildiv(params.n_rows * params.n_cols, index_t(sizeof(bits_t) * 8))),
      bits_d{raft::make_device_vector<bits_t, index_t>(res, n_element)},
      nnz_actual_d{raft::make_device_scalar<index_t>(res, 0)}
  {
  }

  index_t create_bitmap(index_t m, index_t n, float sparsity, std::vector<bits_t>& bitmap)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
    index_t res      = num_ones;

    for (auto& item : bitmap) {
      item = static_cast<bits_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bits_t& element      = bitmap[index / (8 * sizeof(bits_t))];
      index_t bit_position = index % (8 * sizeof(bits_t));

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

    std::vector<bits_t> bits_h(n_element);
    auto stream = raft::resource::get_cuda_stream(res);

    create_bitmap(params.n_rows, params.n_cols, params.sparsity, bits_h);
    update_device(bits_d.data_handle(), bits_h.data(), bits_h.size(), stream);

    resource::sync_stream(res);

    loop_on_state(state, [this]() {
      auto bits_view =
        raft::make_device_vector_view<const bits_t, index_t>(bits_d.data_handle(), bits_d.size());

      index_t max_len   = params.n_rows * params.n_cols;
      auto max_len_view = raft::make_host_scalar_view<index_t>(&max_len);
      auto nnz_actual_view =
        nnz_actual_d.view();  // raft::make_device_scalar_view<index_t>(nnz_actual_d.data_handle());
      raft::popc(this->handle, bits_view, max_len_view, nnz_actual_view);
    });
  }

 private:
  raft::resources res;
  PopcInputs<index_t> params;
  index_t n_element;

  raft::device_vector<bits_t, index_t> bits_d;
  raft::device_scalar<index_t> nnz_actual_d;
};

template <typename index_t>
const std::vector<PopcInputs<index_t>> popc_input_vecs{
  {2, 131072, 0.4},  {8, 131072, 0.5},  {16, 131072, 0.2}, {2, 8192, 0.4},    {16, 8192, 0.5},
  {128, 8192, 0.2},  {1024, 8192, 0.1}, {1024, 8192, 0.1}, {1024, 8192, 0.1}, {1024, 8192, 0.1},

  {1024, 8192, 0.1}, {1024, 8192, 0.1}, {1024, 8192, 0.1}, {1024, 8192, 0.1},

  {1024, 8192, 0.4}, {1024, 8192, 0.5}, {1024, 8192, 0.2}, {1024, 8192, 0.4}, {1024, 8192, 0.5},
  {1024, 8192, 0.2}, {1024, 8192, 0.4}, {1024, 8192, 0.5}, {1024, 8192, 0.2}, {1024, 8192, 0.4},
  {1024, 8192, 0.5}, {1024, 8192, 0.2},

  {1024, 8192, 0.5}, {1024, 8192, 0.2}, {1024, 8192, 0.4}, {1024, 8192, 0.5}, {1024, 8192, 0.2},
  {1024, 8192, 0.4}, {1024, 8192, 0.5}, {1024, 8192, 0.2}};

using PopcBenchI64 = popc_bench<int64_t>;

RAFT_BENCH_REGISTER(PopcBenchI64, "", popc_input_vecs<int64_t>);

}  // namespace raft::bench::core
