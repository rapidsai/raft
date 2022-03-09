/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/spatial/knn/knn.hpp>

#include <raft/random/rng.hpp>
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
struct selection : public Fixture {
  selection(const std::string& name, const params& p) : Fixture(name), params_(p) {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    auto in_len = params_.n_inputs * params_.input_len;
    alloc(in_dists_, in_len, false);
    alloc(in_ids_, in_len, false);
    alloc(out_dists_, params_.n_inputs * params_.k, false);
    alloc(out_ids_, params_.n_inputs * params_.k, false);

    raft::sparse::iota_fill(in_ids_, IdxT(params_.n_inputs), IdxT(params_.input_len), stream);
    raft::random::Rng(42).uniform(in_dists_, in_len, KeyT(-1.0), KeyT(1.0), stream);
  }

  void deallocateBuffers(const ::benchmark::State& state) override
  {
    dealloc(in_dists_, params_.n_inputs * params_.input_len);
    dealloc(in_ids_, params_.n_inputs * params_.input_len);
    dealloc(out_dists_, params_.n_inputs * params_.k);
    dealloc(out_ids_, params_.n_inputs * params_.k);
  }

  void runBenchmark(::benchmark::State& state) override
  {
    rmm::mr::cuda_memory_resource cuda_mr;
    rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{
      &cuda_mr, size_t(1) << size_t(30), size_t(16) << size_t(30)};
    rmm::mr::set_current_device_resource(&pool_mr);
    try {
      std::ostringstream label_stream;
      label_stream << params_.n_inputs << "#" << params_.input_len << "#" << params_.k;
      state.SetLabel(label_stream.str());
      loopOnState(state, [this]() {
        raft::spatial::knn::select_k<IdxT, KeyT>(in_dists_,
                                                 in_ids_,
                                                 params_.n_inputs,
                                                 params_.input_len,
                                                 out_dists_,
                                                 out_ids_,
                                                 params_.select_min,
                                                 params_.k,
                                                 stream,
                                                 Algo);
      });
    } catch (raft::exception& e) {
      state.SkipWithError(e.what());
    }
    rmm::mr::set_current_device_resource(nullptr);
  }

 private:
  params params_;
  KeyT *in_dists_, *out_dists_;
  IdxT *in_ids_, *out_ids_;
};

const std::vector<params> kInputs{
  {10000, 10, 3, true},     {10000, 10, 10, true},     {10000, 700, 3, true},
  {10000, 700, 32, true},   {10000, 2000, 64, true},   {10000, 10000, 7, true},
  {10000, 10000, 19, true}, {10000, 10000, 127, true},

  {1000, 10000, 1, true},   {1000, 10000, 2, true},    {1000, 10000, 4, true},
  {1000, 10000, 8, true},   {1000, 10000, 16, true},   {1000, 10000, 32, true},
  {1000, 10000, 64, true},  {1000, 10000, 128, true},  {1000, 10000, 256, true},
  {1000, 10000, 512, true}, {1000, 10000, 1024, true}, {1000, 10000, 2048, true},

  {100, 100000, 1, true},   {100, 100000, 2, true},    {100, 100000, 4, true},
  {100, 100000, 8, true},   {100, 100000, 16, true},   {100, 100000, 32, true},
  {100, 100000, 64, true},  {100, 100000, 128, true},  {100, 100000, 256, true},
  {100, 100000, 512, true}, {100, 100000, 1024, true}, {100, 100000, 2048, true},

  {10, 1000000, 1, true},   {10, 1000000, 2, true},    {10, 1000000, 4, true},
  {10, 1000000, 8, true},   {10, 1000000, 16, true},   {10, 1000000, 32, true},
  {10, 1000000, 64, true},  {10, 1000000, 128, true},  {10, 1000000, 256, true},
  {10, 1000000, 512, true}, {10, 1000000, 1024, true}, {10, 1000000, 2048, true},
};

#define SELECTION_REGISTER(KeyT, IdxT, Algo)                                      \
  namespace BENCHMARK_PRIVATE_NAME(selection)                                     \
  {                                                                               \
    using SelectK = selection<KeyT, IdxT, raft::spatial::knn::SelectKAlgo::Algo>; \
    RAFT_BENCH_REGISTER(params, SelectK, #KeyT "/" #IdxT "/" #Algo, kInputs);     \
  }

SELECTION_REGISTER(float, int, FAISS);
SELECTION_REGISTER(float, int, RADIX_8_BITS);
SELECTION_REGISTER(float, int, RADIX_11_BITS);
SELECTION_REGISTER(float, int, WARP_SORT);

// SELECTION_REGISTER(double, int, FAISS);
// SELECTION_REGISTER(double, int, RADIX_8_BITS);
// SELECTION_REGISTER(double, int, RADIX_11_BITS);
// SELECTION_REGISTER(double, int, WARP_SORT);

}  // namespace raft::bench::spatial
