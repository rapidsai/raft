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

#include <raft_internal/neighbors/refine_helper.cuh>

#include <common/benchmark.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/detail/refine.cuh>
#include <raft/neighbors/refine.cuh>
#include <raft/random/rng.cuh>

#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.cuh>
#endif

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <iostream>
#include <sstream>

using namespace raft::neighbors;

namespace raft::bench::neighbors {

template <typename IdxT>
inline auto operator<<(std::ostream& os, const RefineInputs<IdxT>& p) -> std::ostream&
{
  os << p.n_rows << "#" << p.dim << "#" << p.n_queries << "#" << p.k0 << "#" << p.k << "#"
     << (p.host_data ? "host" : "device");
  return os;
}

RefineInputs<int64_t> p;

template <typename DataT, typename DistanceT, typename IdxT>
class RefineAnn : public fixture {
 public:
  RefineAnn(RefineInputs<IdxT> p) : data(handle_, p) {}

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << data.p;
    state.SetLabel(label_stream.str());

    auto old_mr = rmm::mr::get_current_device_resource();
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(old_mr);
    rmm::mr::set_current_device_resource(&pool_mr);

    if (data.p.host_data) {
      loop_on_state(state, [this]() {
        raft::neighbors::refine<IdxT, DataT, DistanceT, IdxT>(handle_,
                                                              data.dataset_host.view(),
                                                              data.queries_host.view(),
                                                              data.candidates_host.view(),
                                                              data.refined_indices_host.view(),
                                                              data.refined_distances_host.view(),
                                                              data.p.metric);
      });
    } else {
      loop_on_state(state, [&]() {
        raft::neighbors::refine<IdxT, DataT, DistanceT, IdxT>(handle_,
                                                              data.dataset.view(),
                                                              data.queries.view(),
                                                              data.candidates.view(),
                                                              data.refined_indices.view(),
                                                              data.refined_distances.view(),
                                                              data.p.metric);
      });
    }
    rmm::mr::set_current_device_resource(old_mr);
  }

 private:
  raft::handle_t handle_;
  RefineHelper<DataT, DistanceT, IdxT> data;
};

std::vector<RefineInputs<int64_t>> getInputs()
{
  std::vector<RefineInputs<int64_t>> out;
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded;
  for (bool host_data : {true, false}) {
    for (int64_t n_queries : {1000, 10000}) {
      for (int64_t dim : {128, 512}) {
        out.push_back(RefineInputs<int64_t>{n_queries, 2000000, dim, 32, 128, metric, host_data});
        out.push_back(RefineInputs<int64_t>{n_queries, 2000000, dim, 10, 40, metric, host_data});
      }
    }
  }
  return out;
}

using refine_float_int64 = RefineAnn<float, float, int64_t>;
RAFT_BENCH_REGISTER(refine_float_int64, "", getInputs());

using refine_uint8_int64 = RefineAnn<uint8_t, float, int64_t>;
RAFT_BENCH_REGISTER(refine_uint8_int64, "", getInputs());
}  // namespace raft::bench::neighbors
