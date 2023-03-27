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
#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cudart_utils.hpp>

#if defined RAFT_COMPILED
#include <raft/matrix/specializations.cuh>
#endif

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

template <typename KeyT, typename IdxT, select::Algo Algo>
struct selection : public fixture {
  explicit selection(const select::params& p)
    : params_(p),
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
  }

  void run_benchmark(::benchmark::State& state) override  // NOLINT
  {
    device_resources handle{stream};
    using_pool_memory_res res;
    try {
      std::ostringstream label_stream;
      label_stream << params_.batch_size << "#" << params_.len << "#" << params_.k;
      if (params_.use_same_leading_bits) { label_stream << "#same-leading-bits"; }
      state.SetLabel(label_stream.str());
      loop_on_state(state, [this, &handle]() {
        select::select_k_impl<KeyT, IdxT>(handle,
                                          Algo,
                                          in_dists_.data(),
                                          in_ids_.data(),
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
};

#define SELECTION_REGISTER(KeyT, IdxT, A)                          \
  namespace BENCHMARK_PRIVATE_NAME(selection)                      \
  {                                                                \
    using SelectK = selection<KeyT, IdxT, select::Algo::A>;        \
    RAFT_BENCH_REGISTER(SelectK, #KeyT "/" #IdxT "/" #A, kInputs); \
  }

SELECTION_REGISTER(float, uint32_t, kPublicApi);             // NOLINT
SELECTION_REGISTER(float, uint32_t, kRadix8bits);            // NOLINT
SELECTION_REGISTER(float, uint32_t, kRadix11bits);           // NOLINT
SELECTION_REGISTER(float, uint32_t, kRadix11bitsExtraPass);  // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpAuto);              // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpImmediate);         // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpFiltered);          // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpDistributed);       // NOLINT
SELECTION_REGISTER(float, uint32_t, kWarpDistributedShm);    // NOLINT

SELECTION_REGISTER(double, uint32_t, kRadix8bits);            // NOLINT
SELECTION_REGISTER(double, uint32_t, kRadix11bits);           // NOLINT
SELECTION_REGISTER(double, uint32_t, kRadix11bitsExtraPass);  // NOLINT
SELECTION_REGISTER(double, uint32_t, kWarpAuto);              // NOLINT

SELECTION_REGISTER(double, int64_t, kRadix8bits);            // NOLINT
SELECTION_REGISTER(double, int64_t, kRadix11bits);           // NOLINT
SELECTION_REGISTER(double, int64_t, kRadix11bitsExtraPass);  // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpImmediate);         // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpFiltered);          // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpDistributed);       // NOLINT
SELECTION_REGISTER(double, int64_t, kWarpDistributedShm);    // NOLINT

}  // namespace raft::matrix
