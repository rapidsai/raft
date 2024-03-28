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

#include <raft/core/copy.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/thirdparty/mdspan/include/experimental/mdspan>

#include <cstdint>

namespace raft::bench::core {

template <typename IdxT, std::size_t Rank>
auto constexpr const default_dims = []() {
  auto dims = std::array<IdxT, Rank>{};
  std::fill(dims.begin(), dims.end(), 2);
  return dims;
}();

template <typename IdxT>
auto constexpr const default_dims<IdxT, std::size_t{1}> = std::array<IdxT, 1>{3000000};

template <typename IdxT>
auto constexpr const default_dims<IdxT, std::size_t{2}> = std::array<IdxT, 2>{1000, 3000};

template <typename IdxT>
auto constexpr const default_dims<IdxT, std::size_t{3}> = std::array<IdxT, 3>{20, 300, 500};

template <typename T,
          typename IdxT,
          typename LayoutPolicy,
          memory_type MemType,
          std::size_t Rank,
          typename = std::make_index_sequence<Rank>>
struct bench_array_type;

template <typename T,
          typename IdxT,
          typename LayoutPolicy,
          memory_type MemType,
          std::size_t Rank,
          std::size_t... S>
struct bench_array_type<T, IdxT, LayoutPolicy, MemType, Rank, std::index_sequence<S...>> {
  template <std::size_t>
  auto static constexpr const extent_type = raft::dynamic_extent;

  using type =
    std::conditional_t<MemType == memory_type::host,
                       host_mdarray<T, extents<IdxT, extent_type<S>...>, LayoutPolicy>,
                       device_mdarray<T, extents<IdxT, extent_type<S>...>, LayoutPolicy>>;
};

template <typename SrcT,
          typename DstT,
          typename IdxT,
          typename SrcLayoutPolicy,
          typename DstLayoutPolicy,
          memory_type SrcMemType,
          memory_type DstMemType,
          std::size_t Rank>
struct params {
  std::array<IdxT, Rank> dims = default_dims<IdxT, Rank>;
  using src_array_type =
    typename bench_array_type<SrcT, IdxT, SrcLayoutPolicy, SrcMemType, Rank>::type;
  using dst_array_type =
    typename bench_array_type<DstT, IdxT, DstLayoutPolicy, DstMemType, Rank>::type;
};

template <typename SrcT,
          typename DstT,
          typename IdxT,
          typename SrcLayoutPolicy,
          typename DstLayoutPolicy,
          memory_type SrcMemType,
          memory_type DstMemType,
          std::size_t Rank>
struct CopyBench : public fixture {
  using params_type =
    params<SrcT, DstT, IdxT, SrcLayoutPolicy, DstLayoutPolicy, SrcMemType, DstMemType, Rank>;
  using src_array_type = typename params_type::src_array_type;
  using dst_array_type = typename params_type::dst_array_type;
  explicit CopyBench(const params_type& ps)
    : fixture{true},
      res_{},
      params_{ps},
      src_{
        res_,
        typename src_array_type::mapping_type{
          std::apply([](auto... exts) { return make_extents<IdxT>(exts...); }, ps.dims)},
        typename src_array_type::container_policy_type{},
      },
      dst_{
        res_,
        typename dst_array_type::mapping_type{
          std::apply([](auto... exts) { return make_extents<IdxT>(exts...); }, ps.dims)},
        typename dst_array_type::container_policy_type{},
      }
  {
    res_.get_cublas_handle();  // initialize cublas handle
    auto src_data = std::vector<SrcT>(src_.size());
    std::iota(src_data.begin(), src_data.end(), SrcT{});
    raft::copy(src_.data_handle(), src_data.data(), src_.size(), res_.get_stream());
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() { raft::copy(res_, dst_.view(), src_.view()); });
  }

 private:
  raft::device_resources res_;
  params_type params_;
  src_array_type src_;
  dst_array_type dst_;
};

template <typename ParamsT>
auto static const inputs = std::vector<ParamsT>{ParamsT{}};

#define COPY_REGISTER(BenchT) \
  RAFT_BENCH_REGISTER(BenchT, "BenchT", inputs<typename BenchT::params_type>)

using copy_bench_device_device_1d_same_dtype_same_layout        = CopyBench<int,
                                                                     int,
                                                                     int,
                                                                     layout_c_contiguous,
                                                                     layout_c_contiguous,
                                                                     memory_type::device,
                                                                     memory_type::device,
                                                                     1>;
using copy_bench_device_device_1d_same_dtype_diff_layout        = CopyBench<int,
                                                                     int,
                                                                     int,
                                                                     layout_c_contiguous,
                                                                     layout_f_contiguous,
                                                                     memory_type::device,
                                                                     memory_type::device,
                                                                     1>;
using copy_bench_device_device_1d_diff_dtype_diff_layout        = CopyBench<float,
                                                                     double,
                                                                     int,
                                                                     layout_c_contiguous,
                                                                     layout_f_contiguous,
                                                                     memory_type::device,
                                                                     memory_type::device,
                                                                     1>;
using copy_bench_device_device_2d_same_dtype_diff_layout        = CopyBench<int,
                                                                     int,
                                                                     int,
                                                                     layout_c_contiguous,
                                                                     layout_f_contiguous,
                                                                     memory_type::device,
                                                                     memory_type::device,
                                                                     2>;
using copy_bench_device_device_2d_same_dtype_diff_layout_cublas = CopyBench<float,
                                                                            float,
                                                                            int,
                                                                            layout_c_contiguous,
                                                                            layout_f_contiguous,
                                                                            memory_type::device,
                                                                            memory_type::device,
                                                                            2>;
using copy_bench_device_device_3d_diff_dtype_diff_layout        = CopyBench<float,
                                                                     double,
                                                                     int,
                                                                     layout_c_contiguous,
                                                                     layout_f_contiguous,
                                                                     memory_type::device,
                                                                     memory_type::device,
                                                                     3>;
using copy_bench_device_device_3d_diff_dtype_same_layout        = CopyBench<float,
                                                                     double,
                                                                     int,
                                                                     layout_c_contiguous,
                                                                     layout_c_contiguous,
                                                                     memory_type::device,
                                                                     memory_type::device,
                                                                     3>;

using copy_bench_host_host_1d_same_dtype_same_layout             = CopyBench<int,
                                                                 int,
                                                                 int,
                                                                 layout_c_contiguous,
                                                                 layout_c_contiguous,
                                                                 memory_type::host,
                                                                 memory_type::host,
                                                                 1>;
using copy_bench_host_host_1d_same_dtype_diff_layout             = CopyBench<int,
                                                                 int,
                                                                 int,
                                                                 layout_c_contiguous,
                                                                 layout_f_contiguous,
                                                                 memory_type::host,
                                                                 memory_type::host,
                                                                 1>;
using copy_bench_host_host_1d_diff_dtype_diff_layout             = CopyBench<float,
                                                                 double,
                                                                 int,
                                                                 layout_c_contiguous,
                                                                 layout_f_contiguous,
                                                                 memory_type::host,
                                                                 memory_type::host,
                                                                 1>;
using copy_bench_host_host_2d_same_dtype_diff_layout             = CopyBench<int,
                                                                 int,
                                                                 int,
                                                                 layout_c_contiguous,
                                                                 layout_f_contiguous,
                                                                 memory_type::host,
                                                                 memory_type::host,
                                                                 2>;
using copy_bench_host_host_2d_same_dtype_diff_layout_float_float = CopyBench<float,
                                                                             float,
                                                                             int,
                                                                             layout_c_contiguous,
                                                                             layout_f_contiguous,
                                                                             memory_type::host,
                                                                             memory_type::host,
                                                                             2>;
using copy_bench_host_host_3d_diff_dtype_same_layout             = CopyBench<float,
                                                                 double,
                                                                 int,
                                                                 layout_c_contiguous,
                                                                 layout_c_contiguous,
                                                                 memory_type::host,
                                                                 memory_type::host,
                                                                 3>;
using copy_bench_host_host_3d_diff_dtype_diff_layout             = CopyBench<float,
                                                                 double,
                                                                 int,
                                                                 layout_c_contiguous,
                                                                 layout_f_contiguous,
                                                                 memory_type::host,
                                                                 memory_type::host,
                                                                 3>;

using copy_bench_device_host_1d_same_dtype_same_layout        = CopyBench<int,
                                                                   int,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_c_contiguous,
                                                                   memory_type::device,
                                                                   memory_type::host,
                                                                   1>;
using copy_bench_device_host_1d_same_dtype_diff_layout        = CopyBench<int,
                                                                   int,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::device,
                                                                   memory_type::host,
                                                                   1>;
using copy_bench_device_host_1d_diff_dtype_diff_layout        = CopyBench<float,
                                                                   double,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::device,
                                                                   memory_type::host,
                                                                   1>;
using copy_bench_device_host_2d_same_dtype_diff_layout        = CopyBench<int,
                                                                   int,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::device,
                                                                   memory_type::host,
                                                                   2>;
using copy_bench_device_host_2d_same_dtype_diff_layout_cublas = CopyBench<float,
                                                                          float,
                                                                          int,
                                                                          layout_c_contiguous,
                                                                          layout_f_contiguous,
                                                                          memory_type::device,
                                                                          memory_type::host,
                                                                          2>;
using copy_bench_device_host_3d_diff_dtype_same_layout        = CopyBench<float,
                                                                   double,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_c_contiguous,
                                                                   memory_type::device,
                                                                   memory_type::host,
                                                                   3>;
using copy_bench_device_host_3d_diff_dtype_diff_layout        = CopyBench<float,
                                                                   double,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::device,
                                                                   memory_type::host,
                                                                   3>;

using copy_bench_host_device_1d_same_dtype_same_layout        = CopyBench<int,
                                                                   int,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_c_contiguous,
                                                                   memory_type::host,
                                                                   memory_type::device,
                                                                   1>;
using copy_bench_host_device_1d_same_dtype_diff_layout        = CopyBench<int,
                                                                   int,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::host,
                                                                   memory_type::device,
                                                                   1>;
using copy_bench_host_device_1d_diff_dtype_diff_layout        = CopyBench<float,
                                                                   double,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::host,
                                                                   memory_type::device,
                                                                   1>;
using copy_bench_host_device_2d_same_dtype_diff_layout        = CopyBench<int,
                                                                   int,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::host,
                                                                   memory_type::device,
                                                                   2>;
using copy_bench_host_device_2d_same_dtype_diff_layout_cublas = CopyBench<float,
                                                                          float,
                                                                          int,
                                                                          layout_c_contiguous,
                                                                          layout_f_contiguous,
                                                                          memory_type::host,
                                                                          memory_type::device,
                                                                          2>;
using copy_bench_host_device_3d_diff_dtype_diff_layout        = CopyBench<float,
                                                                   double,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_f_contiguous,
                                                                   memory_type::host,
                                                                   memory_type::device,
                                                                   3>;
using copy_bench_host_device_3d_diff_dtype_same_layout        = CopyBench<float,
                                                                   double,
                                                                   int,
                                                                   layout_c_contiguous,
                                                                   layout_c_contiguous,
                                                                   memory_type::host,
                                                                   memory_type::device,
                                                                   3>;

// COPY_REGISTER(copy_bench_same_dtype_1d_host_host);
COPY_REGISTER(copy_bench_device_device_1d_same_dtype_same_layout);
COPY_REGISTER(copy_bench_device_device_1d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_device_device_1d_diff_dtype_diff_layout);
COPY_REGISTER(copy_bench_device_device_2d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_device_device_2d_same_dtype_diff_layout_cublas);
COPY_REGISTER(copy_bench_device_device_3d_diff_dtype_same_layout);
COPY_REGISTER(copy_bench_device_device_3d_diff_dtype_diff_layout);

COPY_REGISTER(copy_bench_host_host_1d_same_dtype_same_layout);
COPY_REGISTER(copy_bench_host_host_1d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_host_host_1d_diff_dtype_diff_layout);
COPY_REGISTER(copy_bench_host_host_2d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_host_host_2d_same_dtype_diff_layout_float_float);
COPY_REGISTER(copy_bench_host_host_3d_diff_dtype_same_layout);
COPY_REGISTER(copy_bench_host_host_3d_diff_dtype_diff_layout);

COPY_REGISTER(copy_bench_device_host_1d_same_dtype_same_layout);
COPY_REGISTER(copy_bench_device_host_1d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_device_host_1d_diff_dtype_diff_layout);
COPY_REGISTER(copy_bench_device_host_2d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_device_host_2d_same_dtype_diff_layout_cublas);
COPY_REGISTER(copy_bench_device_host_3d_diff_dtype_same_layout);
COPY_REGISTER(copy_bench_device_host_3d_diff_dtype_diff_layout);

COPY_REGISTER(copy_bench_host_device_1d_same_dtype_same_layout);
COPY_REGISTER(copy_bench_host_device_1d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_host_device_1d_diff_dtype_diff_layout);
COPY_REGISTER(copy_bench_host_device_2d_same_dtype_diff_layout);
COPY_REGISTER(copy_bench_host_device_2d_same_dtype_diff_layout_cublas);
COPY_REGISTER(copy_bench_host_device_3d_diff_dtype_same_layout);
COPY_REGISTER(copy_bench_host_device_3d_diff_dtype_diff_layout);

}  // namespace raft::bench::core
