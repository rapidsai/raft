/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/spatial/knn/knn.cuh>

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.cuh>
#endif

#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>

#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::bench::spatial {

struct params {
  int n_inputs;
  int input_len;
  int k;
  int select_min;
};

template <typename KeyT, typename IdxT, raft::spatial::knn::SelectKAlgo Algo>
struct selection : public fixture {
  explicit selection(const params& p)
    : params_(p),
      in_dists_(p.n_inputs * p.input_len, stream),
      in_ids_(p.n_inputs * p.input_len, stream),
      out_dists_(p.n_inputs * p.k, stream),
      out_ids_(p.n_inputs * p.k, stream)
  {
    raft::sparse::iota_fill(in_ids_.data(), IdxT(p.n_inputs), IdxT(p.input_len), stream);
    raft::random::RngState state{42};
    raft::random::uniform(handle, state, in_dists_.data(), in_dists_.size(), KeyT(-1.0), KeyT(1.0));
  }

  void run_benchmark(::benchmark::State& state) override
  {
    using_pool_memory_res res;
    try {
      std::ostringstream label_stream;
      label_stream << params_.n_inputs << "#" << params_.input_len << "#" << params_.k;
      state.SetLabel(label_stream.str());
      loop_on_state(state, [this]() {
        raft::spatial::knn::select_k<IdxT, KeyT>(in_dists_.data(),
                                                 in_ids_.data(),
                                                 params_.n_inputs,
                                                 params_.input_len,
                                                 out_dists_.data(),
                                                 out_ids_.data(),
                                                 params_.select_min,
                                                 params_.k,
                                                 stream,
                                                 Algo);
      });
    } catch (raft::exception& e) {
      state.SkipWithError(e.what());
    }
  }

 private:
  const params params_;
  rmm::device_uvector<KeyT> in_dists_, out_dists_;
  rmm::device_uvector<IdxT> in_ids_, out_ids_;
};

const std::vector<params> kInputs{
  {20000, 500, 1, true},   {20000, 500, 2, true},    {20000, 500, 4, true},
  {20000, 500, 8, true},   {20000, 500, 16, true},   {20000, 500, 32, true},
  {20000, 500, 64, true},  {20000, 500, 128, true},  {20000, 500, 256, true},

  {1000, 10000, 1, true},  {1000, 10000, 2, true},   {1000, 10000, 4, true},
  {1000, 10000, 8, true},  {1000, 10000, 16, true},  {1000, 10000, 32, true},
  {1000, 10000, 64, true}, {1000, 10000, 128, true}, {1000, 10000, 256, true},

  {100, 100000, 1, true},  {100, 100000, 2, true},   {100, 100000, 4, true},
  {100, 100000, 8, true},  {100, 100000, 16, true},  {100, 100000, 32, true},
  {100, 100000, 64, true}, {100, 100000, 128, true}, {100, 100000, 256, true},

  {10, 1000000, 1, true},  {10, 1000000, 2, true},   {10, 1000000, 4, true},
  {10, 1000000, 8, true},  {10, 1000000, 16, true},  {10, 1000000, 32, true},
  {10, 1000000, 64, true}, {10, 1000000, 128, true}, {10, 1000000, 256, true},
};

#define SELECTION_REGISTER(KeyT, IdxT, Algo)                                      \
  namespace BENCHMARK_PRIVATE_NAME(selection)                                     \
  {                                                                               \
    using SelectK = selection<KeyT, IdxT, raft::spatial::knn::SelectKAlgo::Algo>; \
    RAFT_BENCH_REGISTER(SelectK, #KeyT "/" #IdxT "/" #Algo, kInputs);             \
  }

SELECTION_REGISTER(float, int, FAISS);
SELECTION_REGISTER(float, int, RADIX_8_BITS);
SELECTION_REGISTER(float, int, RADIX_11_BITS);
SELECTION_REGISTER(float, int, WARP_SORT);

SELECTION_REGISTER(double, int, FAISS);
SELECTION_REGISTER(double, int, RADIX_8_BITS);
SELECTION_REGISTER(double, int, RADIX_11_BITS);
SELECTION_REGISTER(double, int, WARP_SORT);

SELECTION_REGISTER(double, size_t, FAISS);
SELECTION_REGISTER(double, size_t, RADIX_8_BITS);
SELECTION_REGISTER(double, size_t, RADIX_11_BITS);
SELECTION_REGISTER(double, size_t, WARP_SORT);

}  // namespace raft::bench::spatial
