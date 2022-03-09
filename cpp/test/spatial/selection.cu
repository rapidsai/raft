/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <raft/cudart_utils.h>

#include "../test_utils.h"

#include <raft/sparse/detail/utils.h>
#include <raft/spatial/knn/knn.cuh>
#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.cuh>
#endif

namespace raft::spatial::selection {

using namespace raft;
using namespace raft::sparse;

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec)
{
  for (auto e : vec)
    os << " " << e;
  return os;
}

struct SelectTestSpec {
  int n_inputs;
  int input_len;
  int k;
  int select_min;
};

std::ostream& operator<<(std::ostream& os, const SelectTestSpec& ss)
{
  os << "spec{size: " << ss.input_len << "*" << ss.n_inputs << ", k: " << ss.k;
  os << (ss.select_min ? "; min}" : "; max}");
  return os;
}

template <typename IdxT>
auto gen_simple_ids(int n_inputs, int input_len) -> std::vector<IdxT>
{
  std::vector<IdxT> out(n_inputs * input_len);
  auto s = rmm::cuda_stream_default;
  rmm::device_uvector<IdxT> out_d(out.size(), s);
  iota_fill(out_d.data(), IdxT(n_inputs), IdxT(input_len), s);
  update_host(out.data(), out_d.data(), out.size(), s);
  s.synchronize();
  return out;
}

template <typename KeyT, typename IdxT>
struct SelectInOutSimple {
 public:
  SelectInOutSimple(const SelectTestSpec& spec,
                    const std::vector<KeyT>& in_dists,
                    const std::vector<KeyT>& out_dists,
                    const std::vector<IdxT>& out_ids)
    : in_dists_(in_dists),
      in_ids_(gen_simple_ids<IdxT>(spec.n_inputs, spec.input_len)),
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
struct SelectInOutComputed {
 public:
  SelectInOutComputed(const SelectTestSpec& spec,
                      knn::SelectKAlgo algo,
                      const std::vector<KeyT>& in_dists,
                      const std::optional<std::vector<IdxT>>& in_ids = std::nullopt)
    : in_dists_(in_dists),
      in_ids_(in_ids.value_or(gen_simple_ids<IdxT>(spec.n_inputs, spec.input_len))),
      out_dists_(spec.n_inputs * spec.k),
      out_ids_(spec.n_inputs * spec.k)
  {
    auto stream = rmm::cuda_stream_default;

    rmm::device_uvector<KeyT> in_dists_d(in_dists_.size(), stream);
    rmm::device_uvector<IdxT> in_ids_d(in_ids_.size(), stream);
    rmm::device_uvector<KeyT> out_dists_d(out_dists_.size(), stream);
    rmm::device_uvector<IdxT> out_ids_d(out_ids_.size(), stream);

    update_device(in_dists_d.data(), in_dists_.data(), in_dists_.size(), stream);
    update_device(in_ids_d.data(), in_ids_.data(), in_ids_.size(), stream);

    raft::spatial::knn::select_k<IdxT, KeyT>(in_dists_d.data(),
                                             in_ids_d.data(),
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

    if (algo != knn::SelectKAlgo::WARP_SORT) {
      // knn::SelectKAlgo::WARP_SORT is stable!
      auto p = topk_sort_permutation(out_dists_, out_ids_, spec.k, spec.select_min);
      apply_permutation(out_dists_, p);
      apply_permutation(out_ids_, p);
    }
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
using Params = std::tuple<SelectTestSpec, knn::SelectKAlgo, InOut>;

template <typename KeyT, typename IdxT, template <typename, typename> typename ParamsReader>
class SelectionTest : public testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::ParamsIn> {
 protected:
  const SelectTestSpec spec;
  const knn::SelectKAlgo algo;

  typename ParamsReader<KeyT, IdxT>::InOut ref;
  SelectInOutComputed<KeyT, IdxT> res;

 public:
  explicit SelectionTest(Params<typename ParamsReader<KeyT, IdxT>::InOut> ps)
    : spec(std::get<0>(ps)),
      algo(std::get<1>(ps)),
      ref(std::get<2>(ps)),
      res(spec, algo, ref.get_in_dists(), ref.get_in_ids())
  {
    // std::cout << "dists in: " << ref.get_in_dists() << std::endl;
    // std::cout << "dists ref:" << ref.get_out_dists() << std::endl;
    // std::cout << "dists out:" << res.get_out_dists() << std::endl;

    // std::cout << std::endl;

    // std::cout << "indices in :" << ref.get_in_ids() << std::endl;
    // std::cout << "indices ref:" << ref.get_out_ids() << std::endl;
    // std::cout << "indices out:" << res.get_out_ids() << std::endl;
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
    ASSERT_TRUE(hostArrMatch(ref.get_out_dists().data(),
                             res.get_out_dists().data(),
                             spec.n_inputs * spec.k,
                             Compare<KeyT>()));
    ASSERT_TRUE(hostArrMatch(
      ref.get_out_ids().data(), res.get_out_ids().data(), spec.n_inputs * spec.k, Compare<IdxT>()));
  }
};

auto selection_algos = testing::Values(knn::SelectKAlgo::FAISS,
                                       knn::SelectKAlgo::RADIX_8_BITS,
                                       knn::SelectKAlgo::RADIX_11_BITS,
                                       knn::SelectKAlgo::WARP_SORT);

template <typename KeyT, typename IdxT>
struct params_simple {
  using InOut = SelectInOutSimple<KeyT, IdxT>;
  using Inputs =
    std::tuple<SelectTestSpec, std::vector<KeyT>, std::vector<KeyT>, std::vector<IdxT>>;
  using ParamsIn = std::tuple<Inputs, knn::SelectKAlgo>;

  static auto read(ParamsIn ps) -> Params<InOut>
  {
    auto ins  = std::get<0>(ps);
    auto algo = std::get<1>(ps);
    return std::make_tuple(
      std::get<0>(ins),
      algo,
      SelectInOutSimple<KeyT, IdxT>(
        std::get<0>(ins), std::get<1>(ins), std::get<2>(ins), std::get<3>(ins)));
  }
};

auto inputs_simple_f = testing::Values(
  params_simple<float, int>::Inputs(
    {5, 5, 5, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, int>::Inputs(
    {5, 5, 3, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, int>::Inputs(
    {5, 7, 3, true},
    {5.0, 4.0, 3.0, 2.0, 1.3, 7.5, 19.0, 9.0, 2.0, 3.0, 3.0, 5.0, 6.0, 4.0, 2.0, 3.0, 5.0, 1.0,
     4.0, 1.0, 1.0, 5.0, 7.0, 2.5, 4.0,  7.0, 8.0, 8.0, 1.0, 3.0, 2.0, 5.0, 4.0, 1.1, 1.2},
    {1.3, 2.0, 3.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.5, 4.0, 5.0, 1.0, 1.1, 1.2},
    {4, 3, 2, 1, 2, 3, 3, 5, 6, 2, 3, 0, 0, 5, 6}),
  params_simple<float, int>::Inputs(
    {1, 7, 3, true}, {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {3, 5, 6}),
  params_simple<float, int>::Inputs(
    {1, 7, 3, false}, {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0}, {5.0, 4.0, 3.0}, {2, 4, 1}),
  params_simple<float, int>::Inputs(
    {1, 7, 3, false}, {2.0, 3.0, 5.0, 9.0, 4.0, 9.0, 9.0}, {9.0, 9.0, 9.0}, {3, 5, 6}));

typedef SelectionTest<float, int, params_simple> SimpleFloatInt;
TEST_P(SimpleFloatInt, Run) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        SimpleFloatInt,
                        testing::Combine(inputs_simple_f, selection_algos));

template <knn::SelectKAlgo RefAlgo>
struct with_ref {
  template <typename KeyT, typename IdxT>
  struct params_random {
    using InOut    = SelectInOutComputed<KeyT, IdxT>;
    using ParamsIn = std::tuple<SelectTestSpec, knn::SelectKAlgo>;

    static auto read(ParamsIn ps) -> Params<InOut>
    {
      auto spec = std::get<0>(ps);
      auto algo = std::get<1>(ps);
      std::vector<KeyT> dists(spec.input_len * spec.n_inputs);

      auto s = rmm::cuda_stream_default;
      rmm::device_uvector<KeyT> dists_d(spec.input_len * spec.n_inputs, s);
      raft::random::Rng r(42);
      r.uniform(dists_d.data(), dists_d.size(), KeyT(-1.0), KeyT(1.0), s);
      update_host(dists.data(), dists_d.data(), dists_d.size(), s);
      s.synchronize();

      return std::make_tuple(spec, algo, SelectInOutComputed<KeyT, IdxT>(spec, algo, dists));
    }
  };
};

auto inputs_random_f = testing::Values(SelectTestSpec{20, 700, 8, true},
                                       SelectTestSpec{100, 1700, 17, true},
                                       SelectTestSpec{100, 1700, 31, true},
                                       SelectTestSpec{100, 1700, 32, false},
                                       SelectTestSpec{100, 1700, 33, false},
                                       SelectTestSpec{100, 1700, 63, false},
                                       SelectTestSpec{100, 1700, 64, false},
                                       SelectTestSpec{100, 1700, 65, false},
                                       SelectTestSpec{100, 1700, 255, true},
                                       SelectTestSpec{100, 1700, 256, true},
                                       SelectTestSpec{100, 1700, 511, false},
                                       SelectTestSpec{100, 1700, 512, true},
                                       SelectTestSpec{100, 1700, 1023, false},
                                       SelectTestSpec{100, 1700, 1024, true},
                                       SelectTestSpec{100, 1700, 1700, true});

typedef SelectionTest<float, int, with_ref<knn::SelectKAlgo::FAISS>::params_random>
  ReferencedRandomFloatInt;
TEST_P(ReferencedRandomFloatInt, Run) { run(); }
INSTANTIATE_TEST_CASE_P(SelectionTest,
                        ReferencedRandomFloatInt,
                        testing::Combine(inputs_random_f, selection_algos));

}  // namespace raft::spatial::selection
