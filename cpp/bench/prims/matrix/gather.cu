/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::bench::matrix {

template <typename IdxT>
struct GatherParams {
  IdxT rows, cols, map_length;
  bool host;
};

template <typename IdxT>
inline auto operator<<(std::ostream& os, const GatherParams<IdxT>& p) -> std::ostream&
{
  os << p.rows << "#" << p.cols << "#" << p.map_length << (p.host ? "#host" : "#device");
  return os;
}

template <typename T, typename MapT, typename IdxT, bool Conditional = false>
struct Gather : public fixture {
  Gather(const GatherParams<IdxT>& p)
    : params(p),
      old_mr(rmm::mr::get_current_device_resource()),
      pool_mr(rmm::mr::get_current_device_resource(), 2 * (1ULL << 30)),
      matrix(this->handle),
      map(this->handle),
      out(this->handle),
      stencil(this->handle),
      matrix_h(this->handle)
  {
    rmm::mr::set_current_device_resource(&pool_mr);
  }

  ~Gather() { rmm::mr::set_current_device_resource(old_mr); }

  void allocate_data(const ::benchmark::State& state) override
  {
    matrix  = raft::make_device_matrix<T, IdxT>(handle, params.rows, params.cols);
    map     = raft::make_device_vector<MapT, IdxT>(handle, params.map_length);
    out     = raft::make_device_matrix<T, IdxT>(handle, params.map_length, params.cols);
    stencil = raft::make_device_vector<T, IdxT>(handle, Conditional ? params.map_length : IdxT(0));

    raft::random::RngState rng{1234};
    raft::random::uniform(
      handle, rng, matrix.data_handle(), params.rows * params.cols, T(-1), T(1));
    raft::random::uniformInt(
      handle, rng, map.data_handle(), params.map_length, (MapT)0, (MapT)params.rows);
    if constexpr (Conditional) {
      raft::random::uniform(handle, rng, stencil.data_handle(), params.map_length, T(-1), T(1));
    }

    if (params.host) {
      matrix_h = raft::make_host_matrix<T, IdxT>(handle, params.rows, params.cols);
      raft::copy(matrix_h.data_handle(), matrix.data_handle(), matrix.size(), stream);
    }
    resource::sync_stream(handle, stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      auto matrix_const_view = raft::make_const_mdspan(matrix.view());
      auto map_const_view    = raft::make_const_mdspan(map.view());
      if constexpr (Conditional) {
        auto stencil_const_view = raft::make_const_mdspan(stencil.view());
        auto pred_op            = raft::plug_const_op(T(0.0), raft::greater_op());
        raft::matrix::gather_if(
          handle, matrix_const_view, out.view(), map_const_view, stencil_const_view, pred_op);
      } else {
        if (params.host) {
          raft::matrix::detail::gather(
            handle, make_const_mdspan(matrix_h.view()), map_const_view, out.view());
        } else {
          raft::matrix::gather(handle, matrix_const_view, map_const_view, out.view());
        }
      }
    });
  }

 private:
  GatherParams<IdxT> params;
  rmm::mr::device_memory_resource* old_mr;
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr;
  raft::device_matrix<T, IdxT> matrix, out;
  raft::host_matrix<T, IdxT> matrix_h;
  raft::device_vector<T, IdxT> stencil;
  raft::device_vector<MapT, IdxT> map;
};  // struct Gather

template <typename T, typename MapT, typename IdxT>
using GatherIf = Gather<T, MapT, IdxT, true>;

const std::vector<GatherParams<int64_t>> gather_inputs_i64 =
  raft::util::itertools::product<GatherParams<int64_t>>(
    {1000000}, {10, 20, 50, 100, 200, 500}, {1000, 10000, 100000, 1000000});

RAFT_BENCH_REGISTER((Gather<float, uint32_t, int64_t>), "", gather_inputs_i64);
RAFT_BENCH_REGISTER((Gather<double, uint32_t, int64_t>), "", gather_inputs_i64);
RAFT_BENCH_REGISTER((GatherIf<float, uint32_t, int64_t>), "", gather_inputs_i64);
RAFT_BENCH_REGISTER((GatherIf<double, uint32_t, int64_t>), "", gather_inputs_i64);

auto inputs_host = raft::util::itertools::product<GatherParams<int64_t>>(
  {10000000}, {100}, {1000, 1000000, 10000000}, {true});
RAFT_BENCH_REGISTER((Gather<float, uint32_t, int64_t>), "Host", inputs_host);

}  // namespace raft::bench::matrix
