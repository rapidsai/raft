/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft_internal/matrix/select_k.cuh>

#include <common/benchmark.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cudart_utils.hpp>

#include <raft/matrix/detail/select_radix.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace raft::matrix {
using namespace raft::bench;  // NOLINT

template <typename KeyT>
struct replace_with_mask {
  KeyT replacement;
  int64_t line_length;
  int64_t spared_inputs;
  constexpr auto inline operator()(int64_t offset, KeyT x, uint8_t mask) -> KeyT
  {
    auto i = offset % line_length;
    // don't replace all the inputs, spare a few elements at the beginning of the input
    return (mask && i >= spared_inputs) ? replacement : x;
  }
};

template <typename KeyT, typename IdxT, select::Algo Algo>
struct selection : public fixture {
  explicit selection(const select::params& p)
    : fixture(p.use_memory_pool),
      params_(p),
      in_dists_(p.batch_size * p.len, stream),
      in_ids_(p.batch_size * p.len, stream),
      out_dists_(p.batch_size * p.k, stream),
      out_ids_(p.batch_size * p.k, stream)
  {
    raft::sparse::iota_fill(in_ids_.data(), IdxT(p.batch_size), IdxT(p.len), stream);
    raft::random::RngState state{42};

    KeyT min_value = -1.0;
    KeyT max_value = 1.0;
    if (p.use_same_leading_bits) {
      if constexpr (std::is_same_v<KeyT, float>) {
        uint32_t min_bits = 0x3F800000;  // 1.0
        uint32_t max_bits = 0x3F8000FF;  // 1.00003
        memcpy(&min_value, &min_bits, sizeof(KeyT));
        memcpy(&max_value, &max_bits, sizeof(KeyT));
      } else if constexpr (std::is_same_v<KeyT, double>) {
        uint64_t min_bits = 0x3FF0000000000000;  // 1.0
        uint64_t max_bits = 0x3FF0000FFFFFFFFF;  // 1.000015
        memcpy(&min_value, &min_bits, sizeof(KeyT));
        memcpy(&max_value, &max_bits, sizeof(KeyT));
      }
    }
    raft::random::uniform(handle, state, in_dists_.data(), in_dists_.size(), min_value, max_value);
    if (p.frac_infinities > 0.0) {
      rmm::device_uvector<uint8_t> mask_buf(p.batch_size * p.len, stream);
      auto mask = make_device_vector_view<uint8_t, size_t>(mask_buf.data(), mask_buf.size());
      raft::random::bernoulli(handle, state, mask, p.frac_infinities);
      KeyT bound = p.select_min ? raft::upper_bound<KeyT>() : raft::lower_bound<KeyT>();
      auto mask_in =
        make_device_vector_view<const uint8_t, size_t>(mask_buf.data(), mask_buf.size());
      auto dists_in  = make_device_vector_view<const KeyT>(in_dists_.data(), in_dists_.size());
      auto dists_out = make_device_vector_view<KeyT>(in_dists_.data(), in_dists_.size());
      raft::linalg::map_offset(handle,
                               dists_out,
                               replace_with_mask<KeyT>{bound, int64_t(p.len), int64_t(p.k / 2)},
                               dists_in,
                               mask_in);
    }
  }

  void run_benchmark(::benchmark::State& state) override  // NOLINT
  {
    try {
      std::ostringstream label_stream;
      label_stream << params_.batch_size << "#" << params_.len << "#" << params_.k;
      if (params_.use_same_leading_bits) { label_stream << "#same-leading-bits"; }
      if (params_.frac_infinities > 0) { label_stream << "#infs-" << params_.frac_infinities; }
      state.SetLabel(label_stream.str());
      common::nvtx::range case_scope("%s - %s", state.name().c_str(), label_stream.str().c_str());
      int iter = 0;
      loop_on_state(state, [&iter, this]() {
        common::nvtx::range lap_scope("lap-", iter++);
        select::select_k_impl<KeyT, IdxT>(handle,
                                          Algo,
                                          in_dists_.data(),
                                          params_.use_index_input ? in_ids_.data() : NULL,
                                          params_.batch_size,
                                          params_.len,
                                          params_.k,
                                          out_dists_.data(),
                                          out_ids_.data(),
                                          params_.select_min);
      });
    } catch (raft::exception& e) {
      state.SkipWithError(e.what());
    }
  }

 private:
  const select::params params_;
  rmm::device_uvector<KeyT> in_dists_, out_dists_;
  rmm::device_uvector<IdxT> in_ids_, out_ids_;
};

const std::vector<select::params> kInputs{
  {20000, 500, 1, true},
  {20000, 500, 2, true},
  {20000, 500, 4, true},
  {20000, 500, 8, true},
  {20000, 500, 16, true},
  {20000, 500, 32, true},
  {20000, 500, 64, true},
  {20000, 500, 128, true},
  {20000, 500, 256, true},

  {1000, 10000, 1, true},
  {1000, 10000, 2, true},
  {1000, 10000, 4, true},
  {1000, 10000, 8, true},
  {1000, 10000, 16, true},
  {1000, 10000, 32, true},
  {1000, 10000, 64, true},
  {1000, 10000, 128, true},
  {1000, 10000, 256, true},

  {100, 100000, 1, true},
  {100, 100000, 2, true},
  {100, 100000, 4, true},
  {100, 100000, 8, true},
  {100, 100000, 16, true},
  {100, 100000, 32, true},
  {100, 100000, 64, true},
  {100, 100000, 128, true},
  {100, 100000, 256, true},

  {10, 1000000, 1, true},
  {10, 1000000, 2, true},
  {10, 1000000, 4, true},
  {10, 1000000, 8, true},
  {10, 1000000, 16, true},
  {10, 1000000, 32, true},
  {10, 1000000, 64, true},
  {10, 1000000, 128, true},
  {10, 1000000, 256, true},

  {10, 1000000, 1, true, false, true},
  {10, 1000000, 2, true, false, true},
  {10, 1000000, 4, true, false, true},
  {10, 1000000, 8, true, false, true},
  {10, 1000000, 16, true, false, true},
  {10, 1000000, 32, true, false, true},
  {10, 1000000, 64, true, false, true},
  {10, 1000000, 128, true, false, true},
  {10, 1000000, 256, true, false, true},

  {10, 1000000, 1, true, false, false, true, 0.1},
  {10, 1000000, 16, true, false, false, true, 0.1},
  {10, 1000000, 64, true, false, false, true, 0.1},
  {10, 1000000, 128, true, false, false, true, 0.1},
  {10, 1000000, 256, true, false, false, true, 0.1},

  {10, 1000000, 1, true, false, false, true, 0.9},
  {10, 1000000, 16, true, false, false, true, 0.9},
  {10, 1000000, 64, true, false, false, true, 0.9},
  {10, 1000000, 128, true, false, false, true, 0.9},
  {10, 1000000, 256, true, false, false, true, 0.9},
  {1000, 10000, 1, true, false, false, true, 0.9},
  {1000, 10000, 16, true, false, false, true, 0.9},
  {1000, 10000, 64, true, false, false, true, 0.9},
  {1000, 10000, 128, true, false, false, true, 0.9},
  {1000, 10000, 256, true, false, false, true, 0.9},

  {10, 1000000, 1, true, false, false, true, 1.0},
  {10, 1000000, 16, true, false, false, true, 1.0},
  {10, 1000000, 64, true, false, false, true, 1.0},
  {10, 1000000, 128, true, false, false, true, 1.0},
  {10, 1000000, 256, true, false, false, true, 1.0},
  {1000, 10000, 1, true, false, false, true, 1.0},
  {1000, 10000, 16, true, false, false, true, 1.0},
  {1000, 10000, 64, true, false, false, true, 1.0},
  {1000, 10000, 128, true, false, false, true, 1.0},
  {1000, 10000, 256, true, false, false, true, 1.0},
  {1000, 10000, 256, true, false, false, true, 0.999},
};

#define SELECTION_REGISTER(KeyT, IdxT, A)                        \
  namespace BENCHMARK_PRIVATE_NAME(selection) {                  \
  using SelectK = selection<KeyT, IdxT, select::Algo::A>;        \
  RAFT_BENCH_REGISTER(SelectK, #KeyT "/" #IdxT "/" #A, kInputs); \
  }

SELECTION_REGISTER(float, uint32_t, kPublicApi);              // NOLINT
SELECTION_REGISTER(float, uint32_t, kRadix8bits);             // NOLINT
SELECTION_REGISTER(float, uint32_t, kRadix11bits);            // NOLINT
SELECTION_REGISTER(float, uint32_t, kRadix11bitsExtraPass);   // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpAuto);               // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpImmediate);          // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpFiltered);           // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpDistributed);        // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpDistributedShm);     // NOLINT

SELECTION_REGISTER(double, uint32_t, kRadix8bits);            // NOLINT
SELECTION_REGISTER(double, uint32_t, kRadix11bits);           // NOLINT
SELECTION_REGISTER(double, uint32_t, kRadix11bitsExtraPass);  // NOLINT
SELECTION_REGISTER(double, uint32_t, kWarpAuto);              // NOLINT

SELECTION_REGISTER(double, int64_t, kRadix8bits);             // NOLINT
SELECTION_REGISTER(double, int64_t, kRadix11bits);            // NOLINT
SELECTION_REGISTER(double, int64_t, kRadix11bitsExtraPass);   // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpImmediate);          // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpFiltered);           // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpDistributed);        // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpDistributedShm);     // NOLINT

// For learning a heuristic of which selection algorithm to use, we
// have a couple of additional constraints when generating the dataset:
// 1. We want these benchmarks to be optionally enabled from the commandline -
//  there are thousands of them, and the run-time is non-trivial. This should be opt-in only
// 2. We test out larger k values - that won't work for all algorithms. This requires filtering
// the input parameters per algorithm.
// This makes the code to generate this dataset different from the code above to
// register other benchmarks
#define SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, A, input)                               \
  {                                                                                       \
    using SelectK = selection<KeyT, IdxT, select::Algo::A>;                               \
    std::stringstream name;                                                               \
    name << "SelectKDataset/" << #KeyT "/" #IdxT "/" #A << "/" << input.batch_size << "/" \
         << input.len << "/" << input.k << "/" << input.use_index_input << "/"            \
         << input.use_memory_pool;                                                        \
    auto* b = ::benchmark::internal::RegisterBenchmarkInternal(                           \
      new raft::bench::internal::Fixture<SelectK, select::params>(name.str(), input));    \
    b->UseManualTime();                                                                   \
    b->Unit(benchmark::kMillisecond);                                                     \
  }

const static size_t MAX_MEMORY = 16 * 1024 * 1024 * 1024ULL;

// registers the input for all algorithms
#define SELECTION_REGISTER_INPUT(KeyT, IdxT, input)                            \
  {                                                                            \
    size_t mem = input.batch_size * input.len * (sizeof(KeyT) + sizeof(IdxT)); \
    if (mem < MAX_MEMORY) {                                                    \
      SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kRadix8bits, input)            \
      SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kRadix11bits, input)           \
      SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kRadix11bitsExtraPass, input)  \
      if (input.k <= raft::matrix::detail::select::warpsort::kMaxCapacity) {   \
        SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kWarpImmediate, input)       \
        SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kWarpFiltered, input)        \
        SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kWarpDistributed, input)     \
        SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kWarpDistributedShm, input)  \
      }                                                                        \
      if (input.k <= raft::neighbors::detail::kFaissMaxK<IdxT, KeyT>()) {      \
        SELECTION_REGISTER_ALGO_INPUT(KeyT, IdxT, kFaissBlockSelect, input)    \
      }                                                                        \
    }                                                                          \
  }

void add_select_k_dataset_benchmarks()
{
  // define a uniform grid
  std::vector<select::params> inputs;

  size_t grid_increment = 1;
  std::vector<int> k_vals;
  for (size_t k = 0; k < 13; k += grid_increment) {
    k_vals.push_back(1 << k);
  }
  // Add in values just past the limit for warp/faiss select
  k_vals.push_back(257);
  k_vals.push_back(2049);

  const static bool select_min = true;
  const static bool use_ids    = false;

  for (size_t row = 0; row < 13; row += grid_increment) {
    for (size_t col = 10; col < 28; col += grid_increment) {
      for (auto k : k_vals) {
        inputs.push_back(
          select::params{size_t(1 << row), size_t(1 << col), k, select_min, use_ids});
      }
    }
  }

  // also add in some random values
  std::default_random_engine rng(42);
  std::uniform_real_distribution<> row_dist(0, 13);
  std::uniform_real_distribution<> col_dist(10, 28);
  std::uniform_real_distribution<> k_dist(0, 13);
  for (size_t i = 0; i < 1024; ++i) {
    auto row = static_cast<size_t>(pow(2, row_dist(rng)));
    auto col = static_cast<size_t>(pow(2, col_dist(rng)));
    auto k   = static_cast<int>(pow(2, k_dist(rng)));
    inputs.push_back(select::params{row, col, k, select_min, use_ids});
  }

  for (auto& input : inputs) {
    SELECTION_REGISTER_INPUT(double, int64_t, input);
    SELECTION_REGISTER_INPUT(double, uint32_t, input);
    SELECTION_REGISTER_INPUT(float, int64_t, input);
    SELECTION_REGISTER_INPUT(float, uint32_t, input);
  }

  // also try again without a memory pool to see if there are significant differences
  for (auto input : inputs) {
    input.use_memory_pool = false;
    SELECTION_REGISTER_INPUT(double, int64_t, input);
    SELECTION_REGISTER_INPUT(double, uint32_t, input);
    SELECTION_REGISTER_INPUT(float, int64_t, input);
    SELECTION_REGISTER_INPUT(float, uint32_t, input);
  }
}
}  // namespace raft::matrix
