/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/detail/selection_faiss.cuh>
#include <raft/neighbors/detail/selection_faiss_helpers.cuh>  // kFaissMax
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include "../test_utils.cuh"

#include <raft/sparse/detail/utils.h>
#include <raft/spatial/knn/knn.cuh>

namespace raft::spatial::selection {

using namespace raft;
using namespace raft::sparse;

struct SelectTestSpec {
  int n_inputs;
  int input_len;
  int k;
  int select_min;
  bool use_index_input = true;
};

std::ostream& operator<<(std::ostream& os, const SelectTestSpec& ss)
{
  os << "spec{size: " << ss.input_len << "*" << ss.n_inputs << ", k: " << ss.k;
  os << (ss.select_min ? "; min}" : "; max}");
  return os;
}

template <typename IdxT>
auto gen_simple_ids(int n_inputs, int input_len, const raft::resources& handle) -> std::vector<IdxT>
{
  std::vector<IdxT> out(n_inputs * input_len);
  auto s = resource::get_cuda_stream(handle);
  rmm::device_uvector<IdxT> out_d(out.size(), s);
  iota_fill(out_d.data(), IdxT(n_inputs), IdxT(input_len), s);
  update_host(out.data(), out_d.data(), out.size(), s);
  s.synchronize();
  return out;
}

template <typename KeyT, typename IdxT>
struct SelectInOutSimple {
 public:
  bool not_supported = false;

  SelectInOutSimple(std::shared_ptr<raft::resources> handle,
                    const SelectTestSpec& spec,
                    const std::vector<KeyT>& in_dists,
                    const std::vector<KeyT>& out_dists,
                    const std::vector<IdxT>& out_ids)
    : in_dists_(in_dists),
      in_ids_(gen_simple_ids<IdxT>(spec.n_inputs, spec.input_len, *handle.get())),
      out_dists_(out_dists),
      out_ids_(out_ids),
      handle_(handle)
  {
  }

  auto get_in_dists() -> std::vector<KeyT>& { return in_dists_; }
  auto get_in_ids() -> std::vector<IdxT>& { return in_ids_; }
  auto get_out_dists() -> std::vector<KeyT>& { return out_dists_; }
  auto get_out_ids() -> std::vector<IdxT>& { return out_ids_; }

 private:
  std::shared_ptr<raft::resources> handle_;
  std::vector<KeyT> in_dists_;
  std::vector<IdxT> in_ids_;
  std::vector<KeyT> out_dists_;
  std::vector<IdxT> out_ids_;
};

template <typename KeyT, typename IdxT>
struct SelectInOutComputed {
 public:
  bool not_supported = false;

  SelectInOutComputed(std::shared_ptr<raft::resources> handle,
                      const SelectTestSpec& spec,
                      knn::SelectKAlgo algo,
                      const std::vector<KeyT>& in_dists,
                      const std::optional<std::vector<IdxT>>& in_ids = std::nullopt)
    : handle_(handle),
      in_dists_(in_dists),
      in_ids_(in_ids.value_or(gen_simple_ids<IdxT>(spec.n_inputs, spec.input_len, *handle.get()))),
      out_dists_(spec.n_inputs * spec.k),
      out_ids_(spec.n_inputs * spec.k)

  {
    // check if the size is supported by the algorithm
    switch (algo) {
      case knn::SelectKAlgo::WARP_SORT:
        if (spec.k > raft::matrix::detail::select::warpsort::kMaxCapacity) {
          not_supported = true;
          return;
        }
        break;
      case knn::SelectKAlgo::FAISS:
        if (spec.k > raft::neighbors::detail::kFaissMaxK<IdxT, KeyT>()) {
          not_supported = true;
          return;
        }
        break;
      default: break;
    }

    auto stream = resource::get_cuda_stream(*handle_);

    rmm::device_uvector<KeyT> in_dists_d(in_dists_.size(), stream);
    rmm::device_uvector<IdxT> in_ids_d(in_ids_.size(), stream);
    rmm::device_uvector<KeyT> out_dists_d(out_dists_.size(), stream);
    rmm::device_uvector<IdxT> out_ids_d(out_ids_.size(), stream);

    update_device(in_dists_d.data(), in_dists_.data(), in_dists_.size(), stream);
    update_device(in_ids_d.data(), in_ids_.data(), in_ids_.size(), stream);

    raft::spatial::knn::select_k<IdxT, KeyT>(in_dists_d.data(),
                                             spec.use_index_input ? in_ids_d.data() : nullptr,
                                             spec.n_inputs,
                                             spec.input_len,
                                             out_dists_d.data(),
                                             out_ids_d.data(),
                                             spec.select_min,
                                             spec.k,
                                             stream,
                                             algo);

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
  std::shared_ptr<raft::resources> handle_;
  std::vector<KeyT> in_dists_;
  std::vector<IdxT> in_ids_;
  std::vector<KeyT> out_dists_;
  std::vector<IdxT> out_ids_;

  auto topk_sort_permutation(const std::vector<KeyT>& vec,
                             const std::vector<IdxT>& inds,
                             int k,
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
  void apply_permutation(std::vector<T>& vec, const std::vector<IdxT>& p)
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
using Params =
  std::tuple<SelectTestSpec, knn::SelectKAlgo, InOut, std::shared_ptr<raft::resources>>;

template <typename KeyT, typename IdxT, template <typename, typename> typename ParamsReader>
class SelectionTest : public testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::ParamsIn> {
 protected:
  std::shared_ptr<raft::resources> handle_;
  const SelectTestSpec spec;
  const knn::SelectKAlgo algo;

  typename ParamsReader<KeyT, IdxT>::InOut ref;
  SelectInOutComputed<KeyT, IdxT> res;

 public:
  explicit SelectionTest(Params<typename ParamsReader<KeyT, IdxT>::InOut> ps)
    : handle_(std::get<3>(ps)),
      spec(std::get<0>(ps)),
      algo(std::get<1>(ps)),
      ref(std::get<2>(ps)),
      res(handle_, spec, algo, ref.get_in_dists(), ref.get_in_ids())
  {
  }

  explicit SelectionTest(typename ParamsReader<KeyT, IdxT>::ParamsIn ps)
    : SelectionTest(ParamsReader<KeyT, IdxT>::read(ps))
  {
  }

  SelectionTest()
    : SelectionTest(testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::ParamsIn>::GetParam())
  {
  }

  void run()
  {
    if (ref.not_supported || res.not_supported) { GTEST_SKIP(); }

    ASSERT_TRUE(hostVecMatch(ref.get_out_dists(), res.get_out_dists(), Compare<KeyT>()));
    // If the dists (keys) are the same, different corresponding ids may end up in the selection due
    // to non-deterministic nature of some implementations.
    auto& in_ids   = ref.get_in_ids();
    auto& in_dists = ref.get_in_dists();

    auto compare_ids = [&in_ids, &in_dists](const IdxT& i, const IdxT& j) {
      if (i == j) return true;
      auto ix_i = size_t(std::find(in_ids.begin(), in_ids.end(), i) - in_ids.begin());
      auto ix_j = size_t(std::find(in_ids.begin(), in_ids.end(), j) - in_ids.begin());
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
  using InOut = SelectInOutSimple<KeyT, IdxT>;
  using Inputs =
    std::tuple<SelectTestSpec, std::vector<KeyT>, std::vector<KeyT>, std::vector<IdxT>>;
  using Handle   = std::shared_ptr<raft::resources>;
  using ParamsIn = std::tuple<Inputs, knn::SelectKAlgo, Handle>;

  static auto read(ParamsIn ps) -> Params<InOut>
  {
    auto ins    = std::get<0>(ps);
    auto algo   = std::get<1>(ps);
    auto handle = std::get<2>(ps);
    return std::make_tuple(
      std::get<0>(ins),
      algo,
      SelectInOutSimple<KeyT, IdxT>(
        handle, std::get<0>(ins), std::get<1>(ins), std::get<2>(ins), std::get<3>(ins)),
      handle);
  }
};

auto inputs_simple_f = testing::Values(
  params_simple<float, int>::Inputs(
    {5, 5, 5, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, int>::Inputs(
    {5, 5, 3, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, int>::Inputs(
    {5, 5, 5, true, false},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, int>::Inputs(
    {5, 5, 3, true, false},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, int>::Inputs(
    {5, 7, 3, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.3, 7.5, 19.0, 9.0, 2.0, 3.0, 3.0, 5.0, 6.0, 4.0, 2.0, 3.0, 5.0, 1.0,
     4.0, 1.0, 1.0, 5.0, 7.0, 2.5, 4.0,  7.0, 8.0, 8.0, 1.0, 3.0, 2.0, 5.0, 4.0, 1.1, 1.2},
    {1.3, 2.0, 3.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.5, 4.0, 5.0, 1.0, 1.1, 1.2},
    {4, 3, 2, 1, 2, 3, 3, 5, 6, 2, 3, 0, 0, 5, 6}),
  params_simple<float, int>::Inputs(
    {1, 7, 3, true, true}, {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {3, 5, 6}),
  params_simple<float, int>::Inputs(
    {1, 7, 3, false, false}, {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0}, {5.0, 4.0, 3.0}, {2, 4, 1}),
  params_simple<float, int>::Inputs(
    {1, 7, 3, false, true}, {2.0, 3.0, 5.0, 9.0, 4.0, 9.0, 9.0}, {9.0, 9.0, 9.0}, {3, 5, 6}),
  params_simple<float, int>::Inputs(
    {1, 130, 5, false, true},
    {19, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     1,  2, 1, 2, 1, 2,  1,  2,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4,
     5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 4, 2, 3, 2, 3, 2, 3, 2, 3, 2, 20},
    {20, 19, 18, 17, 16},
    {129, 0, 117, 116, 115}),
  params_simple<float, int>::Inputs(
    {1, 130, 15, false, true},
    {19, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     1,  2, 1, 2, 1, 2,  1,  2,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4,
     5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 4, 2, 3, 2, 3, 2, 3, 2, 3, 2, 20},
    {20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6},
    {129, 0, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105}));

typedef SelectionTest<float, int, params_simple> SimpleFloatInt;
TEST_P(SimpleFloatInt, Run) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        SimpleFloatInt,
                        testing::Combine(inputs_simple_f,
                                         testing::Values(knn::SelectKAlgo::FAISS,
                                                         knn::SelectKAlgo::RADIX_8_BITS,
                                                         knn::SelectKAlgo::RADIX_11_BITS,
                                                         knn::SelectKAlgo::WARP_SORT),
                                         testing::Values(std::make_shared<raft::resources>())));

template <knn::SelectKAlgo RefAlgo>
struct with_ref {
  template <typename KeyT, typename IdxT>
  struct params_random {
    using InOut    = SelectInOutComputed<KeyT, IdxT>;
    using Handle   = std::shared_ptr<raft::resources>;
    using ParamsIn = std::tuple<SelectTestSpec, knn::SelectKAlgo, Handle>;

    static auto read(ParamsIn ps) -> Params<InOut>
    {
      auto spec   = std::get<0>(ps);
      auto algo   = std::get<1>(ps);
      auto handle = std::get<2>(ps);

      std::vector<KeyT> dists(spec.input_len * spec.n_inputs);

      {
        auto s = resource::get_cuda_stream(*handle);
        rmm::device_uvector<KeyT> dists_d(spec.input_len * spec.n_inputs, s);
        raft::random::RngState r(42);
        normal(*(handle.get()), r, dists_d.data(), dists_d.size(), KeyT(10.0), KeyT(100.0));
        update_host(dists.data(), dists_d.data(), dists_d.size(), s);
        s.synchronize();
      }

      return std::make_tuple(
        spec, algo, SelectInOutComputed<KeyT, IdxT>(handle, spec, RefAlgo, dists), handle);
    }
  };
};

auto inputs_random_longlist = testing::Values(SelectTestSpec{1, 130, 15, false},
                                              SelectTestSpec{1, 128, 15, false},
                                              SelectTestSpec{20, 700, 1, true},
                                              SelectTestSpec{20, 700, 2, true},
                                              SelectTestSpec{20, 700, 3, true},
                                              SelectTestSpec{20, 700, 4, true},
                                              SelectTestSpec{20, 700, 5, true},
                                              SelectTestSpec{20, 700, 6, true},
                                              SelectTestSpec{20, 700, 7, true},
                                              SelectTestSpec{20, 700, 8, true},
                                              SelectTestSpec{20, 700, 9, true},
                                              SelectTestSpec{20, 700, 10, true, false},
                                              SelectTestSpec{20, 700, 11, true},
                                              SelectTestSpec{20, 700, 12, true},
                                              SelectTestSpec{20, 700, 16, true},
                                              SelectTestSpec{100, 1700, 17, true},
                                              SelectTestSpec{100, 1700, 31, true, false},
                                              SelectTestSpec{100, 1700, 32, false},
                                              SelectTestSpec{100, 1700, 33, false},
                                              SelectTestSpec{100, 1700, 63, false},
                                              SelectTestSpec{100, 1700, 64, false, false},
                                              SelectTestSpec{100, 1700, 65, false},
                                              SelectTestSpec{100, 1700, 255, true},
                                              SelectTestSpec{100, 1700, 256, true},
                                              SelectTestSpec{100, 1700, 511, false},
                                              SelectTestSpec{100, 1700, 512, true},
                                              SelectTestSpec{100, 1700, 1023, false, false},
                                              SelectTestSpec{100, 1700, 1024, true},
                                              SelectTestSpec{100, 1700, 1700, true});

auto inputs_random_largesize = testing::Values(SelectTestSpec{100, 100000, 1, true},
                                               SelectTestSpec{100, 100000, 2, true},
                                               SelectTestSpec{100, 100000, 3, true, false},
                                               SelectTestSpec{100, 100000, 7, true},
                                               SelectTestSpec{100, 100000, 16, true},
                                               SelectTestSpec{100, 100000, 31, true},
                                               SelectTestSpec{100, 100000, 32, true, false},
                                               SelectTestSpec{100, 100000, 60, true},
                                               SelectTestSpec{100, 100000, 100, true, false},
                                               SelectTestSpec{100, 100000, 200, true},
                                               SelectTestSpec{100000, 100, 100, false},
                                               SelectTestSpec{1, 100000000, 1, true},
                                               SelectTestSpec{1, 100000000, 16, false, false},
                                               SelectTestSpec{1, 100000000, 64, false},
                                               SelectTestSpec{1, 100000000, 128, true, false},
                                               SelectTestSpec{1, 100000000, 256, false, false});

auto inputs_random_largek = testing::Values(SelectTestSpec{100, 100000, 1000, true},
                                            SelectTestSpec{100, 100000, 2000, false},
                                            SelectTestSpec{100, 100000, 100000, true, false},
                                            SelectTestSpec{100, 100000, 2048, false},
                                            SelectTestSpec{100, 100000, 1237, true});

typedef SelectionTest<float, int, with_ref<knn::SelectKAlgo::FAISS>::params_random>
  ReferencedRandomFloatInt;
TEST_P(ReferencedRandomFloatInt, Run) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        ReferencedRandomFloatInt,
                        testing::Combine(inputs_random_longlist,
                                         testing::Values(knn::SelectKAlgo::RADIX_8_BITS,
                                                         knn::SelectKAlgo::RADIX_11_BITS,
                                                         knn::SelectKAlgo::WARP_SORT),
                                         testing::Values(std::make_shared<raft::resources>())));

typedef SelectionTest<double, size_t, with_ref<knn::SelectKAlgo::FAISS>::params_random>
  ReferencedRandomDoubleSizeT;
TEST_P(ReferencedRandomDoubleSizeT, Run) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        ReferencedRandomDoubleSizeT,
                        testing::Combine(inputs_random_longlist,
                                         testing::Values(knn::SelectKAlgo::RADIX_8_BITS,
                                                         knn::SelectKAlgo::RADIX_11_BITS,
                                                         knn::SelectKAlgo::WARP_SORT),
                                         testing::Values(std::make_shared<raft::resources>())));

typedef SelectionTest<double, int, with_ref<knn::SelectKAlgo::FAISS>::params_random>
  ReferencedRandomDoubleInt;
TEST_P(ReferencedRandomDoubleInt, LargeSize) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        ReferencedRandomDoubleInt,
                        testing::Combine(inputs_random_largesize,
                                         testing::Values(knn::SelectKAlgo::WARP_SORT),
                                         testing::Values(std::make_shared<raft::resources>())));

/** TODO: Fix test failure in RAFT CI
 *
 *  SelectionTest/ReferencedRandomFloatSizeT.LargeK/0
 *  Indicices do not match! ref[91628] = 131.359 != res[36504] = 158.438
 *  Actual: false (actual=36504 != expected=91628 @38999;
 *
 *  SelectionTest/ReferencedRandomFloatSizeT.LargeK/1
 *  ERROR: ref[57977] = 58.9079 != res[21973] = 54.9354
 *  Actual: false (actual=21973 != expected=57977 @107999;
 *
 */
typedef SelectionTest<float, size_t, with_ref<knn::SelectKAlgo::RADIX_11_BITS>::params_random>
  ReferencedRandomFloatSizeT;
TEST_P(ReferencedRandomFloatSizeT, LargeK) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        ReferencedRandomFloatSizeT,
                        testing::Combine(inputs_random_largek,
                                         testing::Values(knn::SelectKAlgo::FAISS),
                                         testing::Values(std::make_shared<raft::resources>())));
}  // namespace raft::spatial::selection
