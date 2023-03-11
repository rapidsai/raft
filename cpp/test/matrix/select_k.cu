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

#include "../test_utils.cuh"

#include <raft_internal/matrix/select_k.cuh>

#ifdef RAFT_DISTANCE_COMPILED
#include <raft/matrix/specializations.cuh>
#endif

#include <raft/core/device_resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <algorithm>
#include <numeric>

namespace raft::matrix {

template <typename IdxT>
auto gen_simple_ids(uint32_t batch_size, uint32_t len) -> std::vector<IdxT>
{
  std::vector<IdxT> out(batch_size * len);
  auto s = rmm::cuda_stream_default;
  rmm::device_uvector<IdxT> out_d(out.size(), s);
  sparse::iota_fill(out_d.data(), IdxT(batch_size), IdxT(len), s);
  update_host(out.data(), out_d.data(), out.size(), s);
  s.synchronize();
  return out;
}

template <typename KeyT, typename IdxT>
struct io_simple {
 public:
  bool not_supported = false;

  io_simple(const select::params& spec,
            const std::vector<KeyT>& in_dists,
            const std::vector<KeyT>& out_dists,
            const std::vector<IdxT>& out_ids)
    : in_dists_(in_dists),
      in_ids_(gen_simple_ids<IdxT>(spec.batch_size, spec.len)),
      out_dists_(out_dists),
      out_ids_(out_ids)
  {
  }

  auto get_in_dists() -> std::vector<KeyT>& { return in_dists_; }
  auto get_in_ids() -> std::vector<IdxT>& { return in_ids_; }
  auto get_out_dists() -> std::vector<KeyT>& { return out_dists_; }
  auto get_out_ids() -> std::vector<IdxT>& { return out_ids_; }

 private:
  std::vector<KeyT> in_dists_;
  std::vector<IdxT> in_ids_;
  std::vector<KeyT> out_dists_;
  std::vector<IdxT> out_ids_;
};

template <typename KeyT, typename IdxT>
struct io_computed {
 public:
  bool not_supported = false;

  io_computed(const select::params& spec,
              const select::Algo& algo,
              const std::vector<KeyT>& in_dists,
              const std::optional<std::vector<IdxT>>& in_ids = std::nullopt)
    : in_dists_(in_dists),
      in_ids_(in_ids.value_or(gen_simple_ids<IdxT>(spec.batch_size, spec.len))),
      out_dists_(spec.batch_size * spec.k),
      out_ids_(spec.batch_size * spec.k)
  {
    // check if the size is supported by the algorithm
    switch (algo) {
      case select::Algo::kWarpAuto:
      case select::Algo::kWarpImmediate:
      case select::Algo::kWarpFiltered:
      case select::Algo::kWarpDistributed:
      case select::Algo::kWarpDistributedShm: {
        if (spec.k > raft::matrix::detail::select::warpsort::kMaxCapacity) {
          not_supported = true;
          return;
        }
      } break;
      default: break;
    }

    device_resources handle{};
    auto stream = handle.get_stream();

    rmm::device_uvector<KeyT> in_dists_d(in_dists_.size(), stream);
    rmm::device_uvector<IdxT> in_ids_d(in_ids_.size(), stream);
    rmm::device_uvector<KeyT> out_dists_d(out_dists_.size(), stream);
    rmm::device_uvector<IdxT> out_ids_d(out_ids_.size(), stream);

    update_device(in_dists_d.data(), in_dists_.data(), in_dists_.size(), stream);
    update_device(in_ids_d.data(), in_ids_.data(), in_ids_.size(), stream);

    select::select_k_impl<KeyT, IdxT>(handle,
                                      algo,
                                      in_dists_d.data(),
                                      spec.use_index_input ? in_ids_d.data() : nullptr,
                                      spec.batch_size,
                                      spec.len,
                                      spec.k,
                                      out_dists_d.data(),
                                      out_ids_d.data(),
                                      spec.select_min);

    update_host(out_dists_.data(), out_dists_d.data(), out_dists_.size(), stream);
    update_host(out_ids_.data(), out_ids_d.data(), out_ids_.size(), stream);

    interruptible::synchronize(stream);

    auto p = topk_sort_permutation(out_dists_, out_ids_, spec.k, spec.select_min);
    apply_permutation(out_dists_, p);
    apply_permutation(out_ids_, p);
  }

  auto get_in_dists() -> std::vector<KeyT>& { return in_dists_; }
  auto get_in_ids() -> std::vector<IdxT>& { return in_ids_; }
  auto get_out_dists() -> std::vector<KeyT>& { return out_dists_; }
  auto get_out_ids() -> std::vector<IdxT>& { return out_ids_; }

 private:
  std::vector<KeyT> in_dists_;
  std::vector<IdxT> in_ids_;
  std::vector<KeyT> out_dists_;
  std::vector<IdxT> out_ids_;

  auto topk_sort_permutation(const std::vector<KeyT>& vec,
                             const std::vector<IdxT>& inds,
                             uint32_t k,
                             bool select_min) -> std::vector<IdxT>
  {
    std::vector<IdxT> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    if (select_min) {
      std::sort(p.begin(), p.end(), [&vec, &inds, k](IdxT i, IdxT j) {
        const IdxT ik = i / k;
        const IdxT jk = j / k;
        if (ik == jk) {
          if (vec[i] == vec[j]) { return inds[i] < inds[j]; }
          return vec[i] < vec[j];
        }
        return ik < jk;
      });
    } else {
      std::sort(p.begin(), p.end(), [&vec, &inds, k](IdxT i, IdxT j) {
        const IdxT ik = i / k;
        const IdxT jk = j / k;
        if (ik == jk) {
          if (vec[i] == vec[j]) { return inds[i] < inds[j]; }
          return vec[i] > vec[j];
        }
        return ik < jk;
      });
    }
    return p;
  }

  template <typename T>
  void apply_permutation(std::vector<T>& vec, const std::vector<IdxT>& p)  // NOLINT
  {
    for (auto i = IdxT(vec.size()) - 1; i > 0; i--) {
      auto j = p[i];
      while (j > i)
        j = p[j];
      std::swap(vec[j], vec[i]);
    }
  }
};

template <typename InOut>
using Params = std::tuple<select::params, select::Algo, InOut>;

template <typename KeyT, typename IdxT, template <typename, typename> typename ParamsReader>
struct SelectK  // NOLINT
  : public testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::params_t> {
  const select::params spec;
  const select::Algo algo;
  typename ParamsReader<KeyT, IdxT>::io_t ref;
  io_computed<KeyT, IdxT> res;

  explicit SelectK(Params<typename ParamsReader<KeyT, IdxT>::io_t> ps)
    : spec(std::get<0>(ps)),
      algo(std::get<1>(ps)),                                 // NOLINT
      ref(std::get<2>(ps)),                                  // NOLINT
      res(spec, algo, ref.get_in_dists(), ref.get_in_ids())  // NOLINT
  {
  }

  explicit SelectK(typename ParamsReader<KeyT, IdxT>::params_t ps)
    : SelectK(ParamsReader<KeyT, IdxT>::read(ps))
  {
  }

  SelectK()
    : SelectK(testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::params_t>::GetParam())
  {
  }

  void run()
  {
    if (ref.not_supported || res.not_supported) { GTEST_SKIP(); }
    ASSERT_TRUE(hostVecMatch(ref.get_out_dists(), res.get_out_dists(), Compare<KeyT>()));

    // If the dists (keys) are the same, different corresponding ids may end up in the selection due
    // to non-deterministic nature of some implementations.
    auto& in_ids     = ref.get_in_ids();
    auto& in_dists   = ref.get_in_dists();
    auto compare_ids = [&in_ids, &in_dists](const IdxT& i, const IdxT& j) {
      if (i == j) return true;
      auto ix_i = uint64_t(std::find(in_ids.begin(), in_ids.end(), i) - in_ids.begin());
      auto ix_j = uint64_t(std::find(in_ids.begin(), in_ids.end(), j) - in_ids.begin());
      if (ix_i >= in_ids.size() || ix_j >= in_ids.size()) return false;
      auto dist_i = in_dists[ix_i];
      auto dist_j = in_dists[ix_j];
      if (dist_i == dist_j) return true;
      std::cout << "ERROR: ref[" << ix_i << "] = " << dist_i << " != "
                << "res[" << ix_j << "] = " << dist_j << std::endl;
      return false;
    };
    ASSERT_TRUE(hostVecMatch(ref.get_out_ids(), res.get_out_ids(), compare_ids));
  }
};

template <typename KeyT, typename IdxT>
struct params_simple {
  using io_t = io_simple<KeyT, IdxT>;
  using input_t =
    std::tuple<select::params, std::vector<KeyT>, std::vector<KeyT>, std::vector<IdxT>>;
  using params_t = std::tuple<input_t, select::Algo>;

  static auto read(params_t ps) -> Params<io_t>
  {
    auto ins  = std::get<0>(ps);
    auto algo = std::get<1>(ps);
    return std::make_tuple(
      std::get<0>(ins),
      algo,
      io_simple<KeyT, IdxT>(
        std::get<0>(ins), std::get<1>(ins), std::get<2>(ins), std::get<3>(ins)));
  }
};

auto inputs_simple_f = testing::Values(
  params_simple<float, uint32_t>::input_t(
    {5, 5, 5, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, uint32_t>::input_t(
    {5, 5, 3, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, uint32_t>::input_t(
    {5, 5, 5, true, false},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, uint32_t>::input_t(
    {5, 5, 3, true, false},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, uint32_t>::input_t(
    {5, 7, 3, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.3, 7.5, 19.0, 9.0, 2.0, 3.0, 3.0, 5.0, 6.0, 4.0, 2.0, 3.0, 5.0, 1.0,
     4.0, 1.0, 1.0, 5.0, 7.0, 2.5, 4.0,  7.0, 8.0, 8.0, 1.0, 3.0, 2.0, 5.0, 4.0, 1.1, 1.2},
    {1.3, 2.0, 3.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.5, 4.0, 5.0, 1.0, 1.1, 1.2},
    {4, 3, 2, 1, 2, 3, 3, 5, 6, 2, 3, 0, 0, 5, 6}),
  params_simple<float, uint32_t>::input_t(
    {1, 7, 3, true, true}, {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {3, 5, 6}),
  params_simple<float, uint32_t>::input_t(
    {1, 7, 3, false, false}, {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0}, {5.0, 4.0, 3.0}, {2, 4, 1}),
  params_simple<float, uint32_t>::input_t(
    {1, 7, 3, false, true}, {2.0, 3.0, 5.0, 9.0, 4.0, 9.0, 9.0}, {9.0, 9.0, 9.0}, {3, 5, 6}),
  params_simple<float, uint32_t>::input_t(
    {1, 130, 5, false, true},
    {19, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     1,  2, 1, 2, 1, 2,  1,  2,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4,
     5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 4, 2, 3, 2, 3, 2, 3, 2, 3, 2, 20},
    {20, 19, 18, 17, 16},
    {129, 0, 117, 116, 115}),
  params_simple<float, uint32_t>::input_t(
    {1, 130, 15, false, true},
    {19, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     1,  2, 1, 2, 1, 2,  1,  2,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4,
     5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 4, 2, 3, 2, 3, 2, 3, 2, 3, 2, 20},
    {20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6},
    {129, 0, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105}));

using SimpleFloatInt = SelectK<float, uint32_t, params_simple>;
TEST_P(SimpleFloatInt, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                // NOLINT
  SelectK,
  SimpleFloatInt,
  testing::Combine(inputs_simple_f,
                   testing::Values(select::Algo::kPublicApi,
                                   select::Algo::kRadix8bits,
                                   select::Algo::kRadix11bits,
                                   select::Algo::kWarpImmediate,
                                   select::Algo::kWarpFiltered,
                                   select::Algo::kWarpDistributed)));

template <select::Algo RefAlgo>
struct with_ref {
  template <typename KeyT, typename IdxT>
  struct params_random {
    using io_t     = io_computed<KeyT, IdxT>;
    using params_t = std::tuple<select::params, select::Algo>;

    static auto read(params_t ps) -> Params<io_t>
    {
      auto spec = std::get<0>(ps);
      auto algo = std::get<1>(ps);
      std::vector<KeyT> dists(spec.len * spec.batch_size);

      raft::device_resources handle;
      {
        auto s = handle.get_stream();
        rmm::device_uvector<KeyT> dists_d(spec.len * spec.batch_size, s);
        raft::random::RngState r(42);
        normal(handle, r, dists_d.data(), dists_d.size(), KeyT(10.0), KeyT(100.0));
        update_host(dists.data(), dists_d.data(), dists_d.size(), s);
        s.synchronize();
      }

      return std::make_tuple(spec, algo, io_computed<KeyT, IdxT>(spec, RefAlgo, dists));
    }
  };
};

auto inputs_random_longlist = testing::Values(select::params{1, 130, 15, false},
                                              select::params{1, 128, 15, false},
                                              select::params{20, 700, 1, true},
                                              select::params{20, 700, 2, true},
                                              select::params{20, 700, 3, true},
                                              select::params{20, 700, 4, true},
                                              select::params{20, 700, 5, true},
                                              select::params{20, 700, 6, true},
                                              select::params{20, 700, 7, true},
                                              select::params{20, 700, 8, true},
                                              select::params{20, 700, 9, true},
                                              select::params{20, 700, 10, true, false},
                                              select::params{20, 700, 11, true},
                                              select::params{20, 700, 12, true},
                                              select::params{20, 700, 16, true},
                                              select::params{100, 1700, 17, true},
                                              select::params{100, 1700, 31, true, false},
                                              select::params{100, 1700, 32, false},
                                              select::params{100, 1700, 33, false},
                                              select::params{100, 1700, 63, false},
                                              select::params{100, 1700, 64, false, false},
                                              select::params{100, 1700, 65, false},
                                              select::params{100, 1700, 255, true},
                                              select::params{100, 1700, 256, true},
                                              select::params{100, 1700, 511, false},
                                              select::params{100, 1700, 512, true},
                                              select::params{100, 1700, 1023, false, false},
                                              select::params{100, 1700, 1024, true},
                                              select::params{100, 1700, 1700, true});

auto inputs_random_largesize = testing::Values(select::params{100, 100000, 1, true},
                                               select::params{100, 100000, 2, true},
                                               select::params{100, 100000, 3, true, false},
                                               select::params{100, 100000, 7, true},
                                               select::params{100, 100000, 16, true},
                                               select::params{100, 100000, 31, true},
                                               select::params{100, 100000, 32, true, false},
                                               select::params{100, 100000, 60, true},
                                               select::params{100, 100000, 100, true, false},
                                               select::params{100, 100000, 200, true},
                                               select::params{100000, 100, 100, false},
                                               select::params{1, 1000000000, 1, true},
                                               select::params{1, 1000000000, 16, false, false},
                                               select::params{1, 1000000000, 64, false},
                                               select::params{1, 1000000000, 128, true, false},
                                               select::params{1, 1000000000, 256, false, false});

auto inputs_random_largek = testing::Values(select::params{100, 100000, 1000, true},
                                            select::params{100, 100000, 2000, true},
                                            select::params{100, 100000, 100000, true, false},
                                            select::params{100, 100000, 2048, false},
                                            select::params{100, 100000, 1237, true});

using ReferencedRandomFloatInt =
  SelectK<float, uint32_t, with_ref<select::Algo::kPublicApi>::params_random>;
TEST_P(ReferencedRandomFloatInt, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                          // NOLINT
  SelectK,
  ReferencedRandomFloatInt,
  testing::Combine(inputs_random_longlist,
                   testing::Values(select::Algo::kRadix8bits,
                                   select::Algo::kRadix11bits,
                                   select::Algo::kWarpImmediate,
                                   select::Algo::kWarpFiltered,
                                   select::Algo::kWarpDistributed,
                                   select::Algo::kWarpDistributedShm)));

using ReferencedRandomDoubleSizeT =
  SelectK<double, uint64_t, with_ref<select::Algo::kPublicApi>::params_random>;
TEST_P(ReferencedRandomDoubleSizeT, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                             // NOLINT
  SelectK,
  ReferencedRandomDoubleSizeT,
  testing::Combine(inputs_random_longlist,
                   testing::Values(select::Algo::kRadix8bits,
                                   select::Algo::kRadix11bits,
                                   select::Algo::kWarpImmediate,
                                   select::Algo::kWarpFiltered,
                                   select::Algo::kWarpDistributed,
                                   select::Algo::kWarpDistributedShm)));

using ReferencedRandomDoubleInt =
  SelectK<double, uint32_t, with_ref<select::Algo::kRadix11bits>::params_random>;
TEST_P(ReferencedRandomDoubleInt, LargeSize) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                                 // NOLINT
  SelectK,
  ReferencedRandomDoubleInt,
  testing::Combine(inputs_random_largesize, testing::Values(select::Algo::kWarpAuto)));

using ReferencedRandomFloatSizeT =
  SelectK<float, uint64_t, with_ref<select::Algo::kRadix8bits>::params_random>;
TEST_P(ReferencedRandomFloatSizeT, LargeK) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(SelectK,                       // NOLINT
                        ReferencedRandomFloatSizeT,
                        testing::Combine(inputs_random_largek,
                                         testing::Values(select::Algo::kRadix11bits)));

}  // namespace raft::matrix
