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

#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>

#include <raft/core/operators.cuh>
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/distance/detail/coo_spmv.cuh>

#include "../test_utils.cuh"

#include <type_traits>

namespace raft {
namespace sparse {
namespace distance {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct InputConfiguration {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_t> out_dists_ref_h;

  raft::distance::DistanceType metric;

  float metric_arg = 0.0;
};

using dense_smem_strategy_t = detail::dense_smem_strategy<int, float, 1024>;
using hash_strategy_t       = detail::hash_strategy<int, float, 1024>;

template <typename value_idx, typename value_t, typename strategy_t>
struct SparseDistanceCOOSPMVInputs {
  InputConfiguration<value_idx, value_t> input_configuration;

  float capacity_threshold = 0.5;
  int map_size             = detail::hash_strategy<value_idx, value_t, 1024>::get_map_size();
};

template <typename value_idx, typename value_t, typename strategy_t>
::std::ostream& operator<<(::std::ostream& os,
                           const SparseDistanceCOOSPMVInputs<value_idx, value_t, strategy_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t, typename strategy_t>
class SparseDistanceCOOSPMVTest
  : public ::testing::TestWithParam<SparseDistanceCOOSPMVInputs<value_idx, value_t, strategy_t>> {
 public:
  SparseDistanceCOOSPMVTest()
    : dist_config(handle),
      indptr(0, resource::get_cuda_stream(handle)),
      indices(0, resource::get_cuda_stream(handle)),
      data(0, resource::get_cuda_stream(handle)),
      out_dists(0, resource::get_cuda_stream(handle)),
      out_dists_ref(0, resource::get_cuda_stream(handle))
  {
  }

  template <typename U, std::enable_if_t<std::is_same_v<U, hash_strategy_t>>* = nullptr>
  U make_strategy()
  {
    return strategy_t(dist_config, params.capacity_threshold, params.map_size);
  }

  template <typename U, std::enable_if_t<std::is_same_v<U, dense_smem_strategy_t>>* = nullptr>
  U make_strategy()
  {
    return strategy_t(dist_config);
  }

  template <typename reduce_f, typename accum_f, typename write_f>
  void compute_dist(reduce_f reduce_func, accum_f accum_func, write_f write_func, bool rev = true)
  {
    rmm::device_uvector<value_idx> coo_rows(max(dist_config.b_nnz, dist_config.a_nnz),
                                            resource::get_cuda_stream(dist_config.handle));

    raft::sparse::convert::csr_to_coo(dist_config.b_indptr,
                                      dist_config.b_nrows,
                                      coo_rows.data(),
                                      dist_config.b_nnz,
                                      resource::get_cuda_stream(dist_config.handle));

    strategy_t selected_strategy = make_strategy<strategy_t>();
    detail::balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(out_dists.data(),
                                                                       dist_config,
                                                                       coo_rows.data(),
                                                                       reduce_func,
                                                                       accum_func,
                                                                       write_func,
                                                                       selected_strategy);

    if (rev) {
      raft::sparse::convert::csr_to_coo(dist_config.a_indptr,
                                        dist_config.a_nrows,
                                        coo_rows.data(),
                                        dist_config.a_nnz,
                                        resource::get_cuda_stream(dist_config.handle));

      detail::balanced_coo_pairwise_generalized_spmv_rev<value_idx, value_t>(out_dists.data(),
                                                                             dist_config,
                                                                             coo_rows.data(),
                                                                             reduce_func,
                                                                             accum_func,
                                                                             write_func,
                                                                             selected_strategy);
    }
  }

  void run_spmv()
  {
    switch (params.input_configuration.metric) {
      case raft::distance::DistanceType::InnerProduct:
        compute_dist(raft::mul_op(), raft::add_op(), raft::atomic_add_op(), true);
        break;
      case raft::distance::DistanceType::L2Unexpanded:
        compute_dist(raft::sqdiff_op(), raft::add_op(), raft::atomic_add_op());
        break;
      case raft::distance::DistanceType::Canberra:
        compute_dist(
          [] __device__(value_t a, value_t b) { return fabsf(a - b) / (fabsf(a) + fabsf(b)); },
          raft::add_op(),
          raft::atomic_add_op());
        break;
      case raft::distance::DistanceType::L1:
        compute_dist(absdiff_op(), raft::add_op(), raft::atomic_add_op());
        break;
      case raft::distance::DistanceType::Linf:
        compute_dist(absdiff_op(), raft::max_op(), raft::atomic_max_op());
        break;
      case raft::distance::DistanceType::LpUnexpanded: {
        compute_dist(
          raft::compose_op(raft::pow_const_op<value_t>(params.input_configuration.metric_arg),
                           raft::sub_op()),
          raft::add_op(),
          raft::atomic_add_op());
        value_t p = value_t{1} / params.input_configuration.metric_arg;
        raft::linalg::unaryOp<value_t>(out_dists.data(),
                                       out_dists.data(),
                                       dist_config.a_nrows * dist_config.b_nrows,
                                       raft::pow_const_op<value_t>{p},
                                       resource::get_cuda_stream(dist_config.handle));

      } break;
      default: throw raft::exception("Unknown distance");
    }
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.input_configuration.indptr_h;
    std::vector<value_idx> indices_h = params.input_configuration.indices_h;
    std::vector<value_t> data_h      = params.input_configuration.data_h;

    auto stream = resource::get_cuda_stream(handle);
    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);

    std::vector<value_t> out_dists_ref_h = params.input_configuration.out_dists_ref_h;

    out_dists_ref.resize((indptr_h.size() - 1) * (indptr_h.size() - 1), stream);

    update_device(out_dists_ref.data(), out_dists_ref_h.data(), out_dists_ref_h.size(), stream);
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<
      SparseDistanceCOOSPMVInputs<value_idx, value_t, strategy_t>>::GetParam();

    make_data();

    dist_config.b_nrows   = params.input_configuration.indptr_h.size() - 1;
    dist_config.b_ncols   = params.input_configuration.n_cols;
    dist_config.b_nnz     = params.input_configuration.indices_h.size();
    dist_config.b_indptr  = indptr.data();
    dist_config.b_indices = indices.data();
    dist_config.b_data    = data.data();
    dist_config.a_nrows   = params.input_configuration.indptr_h.size() - 1;
    dist_config.a_ncols   = params.input_configuration.n_cols;
    dist_config.a_nnz     = params.input_configuration.indices_h.size();
    dist_config.a_indptr  = indptr.data();
    dist_config.a_indices = indices.data();
    dist_config.a_data    = data.data();

    int out_size = dist_config.a_nrows * dist_config.b_nrows;

    out_dists.resize(out_size, resource::get_cuda_stream(handle));

    run_spmv();

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle)));
  }

  void compare()
  {
    ASSERT_TRUE(devArrMatch(out_dists_ref.data(),
                            out_dists.data(),
                            params.input_configuration.out_dists_ref_h.size(),
                            CompareApprox<value_t>(1e-3)));
  }

 protected:
  raft::resources handle;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  // output data
  rmm::device_uvector<value_t> out_dists, out_dists_ref;

  raft::sparse::distance::detail::distances_config_t<value_idx, value_t> dist_config;

  SparseDistanceCOOSPMVInputs<value_idx, value_t, strategy_t> params;
};

const InputConfiguration<int, float> input_inner_product = {
  2,
  {0, 2, 4, 6, 8},
  {0, 1, 0, 1, 0, 1, 0, 1},
  {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f},
  {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
  raft::distance::DistanceType::InnerProduct,
  0.0};

const InputConfiguration<int, float> input_l2_unexpanded = {
  2,
  {0, 2, 4, 6, 8},
  {0, 1, 0, 1, 0, 1, 0, 1},  // indices
  {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f},
  {
    // dense output
    0.0,
    4.0,
    3026.0,
    226.0,
    4.0,
    0.0,
    2930.0,
    234.0,
    3026.0,
    2930.0,
    0.0,
    1832.0,
    226.0,
    234.0,
    1832.0,
    0.0,
  },
  raft::distance::DistanceType::L2Unexpanded,
  0.0};

const InputConfiguration<int, float> input_canberra = {
  10,
  {0, 5, 11, 15, 20, 27, 32, 36, 43, 47, 50},
  {0, 1, 3, 6, 8, 0, 1, 2, 3, 5, 6, 1, 2, 4, 8, 0, 2, 3, 4, 7, 0, 1, 2, 3, 4,
   6, 8, 0, 1, 2, 5, 7, 1, 5, 8, 9, 0, 1, 2, 5, 6, 8, 9, 2, 4, 5, 7, 0, 3, 9},  // indices
  {0.5438, 0.2695, 0.4377, 0.7174, 0.9251, 0.7648, 0.3322, 0.7279, 0.4131, 0.5167,
   0.8655, 0.0730, 0.0291, 0.9036, 0.7988, 0.5019, 0.7663, 0.2190, 0.8206, 0.3625,
   0.0411, 0.3995, 0.5688, 0.7028, 0.8706, 0.3199, 0.4431, 0.0535, 0.2225, 0.8853,
   0.1932, 0.3761, 0.3379, 0.1771, 0.2107, 0.228,  0.5279, 0.4885, 0.3495, 0.5079,
   0.2325, 0.2331, 0.3018, 0.6231, 0.2645, 0.8429, 0.6625, 0.0797, 0.2724, 0.4218},
  {0.0,
   3.3954660629919076,
   5.6469232737388815,
   6.373112846266441,
   4.0212880272531715,
   6.916281504639404,
   5.741508386786526,
   5.411470999663036,
   9.0,
   4.977014354725805,
   3.3954660629919076,
   0.0,
   7.56256082439209,
   5.540261147481582,
   4.832322929216881,
   4.62003193872216,
   6.498056792320361,
   4.309846252268695,
   6.317531174829905,
   6.016362684141827,
   5.6469232737388815,
   7.56256082439209,
   0.0,
   5.974878731322299,
   4.898357301336036,
   6.442097410320605,
   5.227077347287883,
   7.134101195584642,
   5.457753923371659,
   7.0,
   6.373112846266441,
   5.540261147481582,
   5.974878731322299,
   0.0,
   5.5507273748583,
   4.897749658726415,
   9.0,
   8.398776718824767,
   3.908281400328807,
   4.83431066343688,
   4.0212880272531715,
   4.832322929216881,
   4.898357301336036,
   5.5507273748583,
   0.0,
   6.632989819428174,
   7.438852294822894,
   5.6631570310967465,
   7.579428202635459,
   6.760811985364303,
   6.916281504639404,
   4.62003193872216,
   6.442097410320605,
   4.897749658726415,
   6.632989819428174,
   0.0,
   5.249404187382862,
   6.072559523278559,
   4.07661278488929,
   6.19678948003145,
   5.741508386786526,
   6.498056792320361,
   5.227077347287883,
   9.0,
   7.438852294822894,
   5.249404187382862,
   0.0,
   3.854811639654704,
   6.652724827169063,
   5.298236851430971,
   5.411470999663036,
   4.309846252268695,
   7.134101195584642,
   8.398776718824767,
   5.6631570310967465,
   6.072559523278559,
   3.854811639654704,
   0.0,
   7.529184598969917,
   6.903282911791188,
   9.0,
   6.317531174829905,
   5.457753923371659,
   3.908281400328807,
   7.579428202635459,
   4.07661278488929,
   6.652724827169063,
   7.529184598969917,
   0.0,
   7.0,
   4.977014354725805,
   6.016362684141827,
   7.0,
   4.83431066343688,
   6.760811985364303,
   6.19678948003145,
   5.298236851430971,
   6.903282911791188,
   7.0,
   0.0},
  raft::distance::DistanceType::Canberra,
  0.0};

const InputConfiguration<int, float> input_lp_unexpanded = {
  10,
  {0, 5, 11, 15, 20, 27, 32, 36, 43, 47, 50},
  {0, 1, 3, 6, 8, 0, 1, 2, 3, 5, 6, 1, 2, 4, 8, 0, 2, 3, 4, 7, 0, 1, 2, 3, 4,
   6, 8, 0, 1, 2, 5, 7, 1, 5, 8, 9, 0, 1, 2, 5, 6, 8, 9, 2, 4, 5, 7, 0, 3, 9},  // indices
  {0.5438, 0.2695, 0.4377, 0.7174, 0.9251, 0.7648, 0.3322, 0.7279, 0.4131, 0.5167,
   0.8655, 0.0730, 0.0291, 0.9036, 0.7988, 0.5019, 0.7663, 0.2190, 0.8206, 0.3625,
   0.0411, 0.3995, 0.5688, 0.7028, 0.8706, 0.3199, 0.4431, 0.0535, 0.2225, 0.8853,
   0.1932, 0.3761, 0.3379, 0.1771, 0.2107, 0.228,  0.5279, 0.4885, 0.3495, 0.5079,
   0.2325, 0.2331, 0.3018, 0.6231, 0.2645, 0.8429, 0.6625, 0.0797, 0.2724, 0.4218},
  {0.0,
   1.31462855332296,
   1.3690307816129905,
   1.698603990921237,
   1.3460470789553531,
   1.6636670712582544,
   1.2651744044972217,
   1.1938329352055201,
   1.8811409082590185,
   1.3653115050624267,
   1.31462855332296,
   0.0,
   1.9447722703291133,
   1.42818777206562,
   1.4685491458946494,
   1.3071999866010466,
   1.4988622861692171,
   0.9698559287406783,
   1.4972023224597841,
   1.5243383567266802,
   1.3690307816129905,
   1.9447722703291133,
   0.0,
   1.2748400840107568,
   1.0599569946448246,
   1.546591282841402,
   1.147526531928459,
   1.447002179128145,
   1.5982242387673176,
   1.3112533607072414,
   1.698603990921237,
   1.42818777206562,
   1.2748400840107568,
   0.0,
   1.038121552545461,
   1.011788365364402,
   1.3907391109256988,
   1.3128200942311496,
   1.19595706584447,
   1.3233328139624725,
   1.3460470789553531,
   1.4685491458946494,
   1.0599569946448246,
   1.038121552545461,
   0.0,
   1.3642741698145529,
   1.3493868683808095,
   1.394942694628328,
   1.572881849642552,
   1.380122665319464,
   1.6636670712582544,
   1.3071999866010466,
   1.546591282841402,
   1.011788365364402,
   1.3642741698145529,
   0.0,
   1.018961640373018,
   1.0114394258945634,
   0.8338711034820684,
   1.1247823842299223,
   1.2651744044972217,
   1.4988622861692171,
   1.147526531928459,
   1.3907391109256988,
   1.3493868683808095,
   1.018961640373018,
   0.0,
   0.7701238110357329,
   1.245486437864406,
   0.5551259549534626,
   1.1938329352055201,
   0.9698559287406783,
   1.447002179128145,
   1.3128200942311496,
   1.394942694628328,
   1.0114394258945634,
   0.7701238110357329,
   0.0,
   1.1886800117391216,
   1.0083692448135637,
   1.8811409082590185,
   1.4972023224597841,
   1.5982242387673176,
   1.19595706584447,
   1.572881849642552,
   0.8338711034820684,
   1.245486437864406,
   1.1886800117391216,
   0.0,
   1.3661374102525012,
   1.3653115050624267,
   1.5243383567266802,
   1.3112533607072414,
   1.3233328139624725,
   1.380122665319464,
   1.1247823842299223,
   0.5551259549534626,
   1.0083692448135637,
   1.3661374102525012,
   0.0},
  raft::distance::DistanceType::LpUnexpanded,
  2.0};

const InputConfiguration<int, float> input_linf = {
  10,
  {0, 5, 11, 15, 20, 27, 32, 36, 43, 47, 50},
  {0, 1, 3, 6, 8, 0, 1, 2, 3, 5, 6, 1, 2, 4, 8, 0, 2, 3, 4, 7, 0, 1, 2, 3, 4,
   6, 8, 0, 1, 2, 5, 7, 1, 5, 8, 9, 0, 1, 2, 5, 6, 8, 9, 2, 4, 5, 7, 0, 3, 9},  // indices
  {0.5438, 0.2695, 0.4377, 0.7174, 0.9251, 0.7648, 0.3322, 0.7279, 0.4131, 0.5167,
   0.8655, 0.0730, 0.0291, 0.9036, 0.7988, 0.5019, 0.7663, 0.2190, 0.8206, 0.3625,
   0.0411, 0.3995, 0.5688, 0.7028, 0.8706, 0.3199, 0.4431, 0.0535, 0.2225, 0.8853,
   0.1932, 0.3761, 0.3379, 0.1771, 0.2107, 0.228,  0.5279, 0.4885, 0.3495, 0.5079,
   0.2325, 0.2331, 0.3018, 0.6231, 0.2645, 0.8429, 0.6625, 0.0797, 0.2724, 0.4218},
  {0.0,
   0.9251771844789913,
   0.9036452083899731,
   0.9251771844789913,
   0.8706483735804971,
   0.9251771844789913,
   0.717493881903289,
   0.6920214832303888,
   0.9251771844789913,
   0.9251771844789913,
   0.9251771844789913,
   0.0,
   0.9036452083899731,
   0.8655339692155823,
   0.8706483735804971,
   0.8655339692155823,
   0.8655339692155823,
   0.6329837991017668,
   0.8655339692155823,
   0.8655339692155823,
   0.9036452083899731,
   0.9036452083899731,
   0.0,
   0.7988276152181608,
   0.7028075145996631,
   0.9036452083899731,
   0.9036452083899731,
   0.9036452083899731,
   0.8429599432532096,
   0.9036452083899731,
   0.9251771844789913,
   0.8655339692155823,
   0.7988276152181608,
   0.0,
   0.48376552205293305,
   0.8206394616536681,
   0.8206394616536681,
   0.8206394616536681,
   0.8429599432532096,
   0.8206394616536681,
   0.8706483735804971,
   0.8706483735804971,
   0.7028075145996631,
   0.48376552205293305,
   0.0,
   0.8706483735804971,
   0.8706483735804971,
   0.8706483735804971,
   0.8429599432532096,
   0.8706483735804971,
   0.9251771844789913,
   0.8655339692155823,
   0.9036452083899731,
   0.8206394616536681,
   0.8706483735804971,
   0.0,
   0.8853924473642432,
   0.535821510936138,
   0.6497196601457607,
   0.8853924473642432,
   0.717493881903289,
   0.8655339692155823,
   0.9036452083899731,
   0.8206394616536681,
   0.8706483735804971,
   0.8853924473642432,
   0.0,
   0.5279604218147174,
   0.6658348373853169,
   0.33799874888632914,
   0.6920214832303888,
   0.6329837991017668,
   0.9036452083899731,
   0.8206394616536681,
   0.8706483735804971,
   0.535821510936138,
   0.5279604218147174,
   0.0,
   0.662579808115858,
   0.5079750812968089,
   0.9251771844789913,
   0.8655339692155823,
   0.8429599432532096,
   0.8429599432532096,
   0.8429599432532096,
   0.6497196601457607,
   0.6658348373853169,
   0.662579808115858,
   0.0,
   0.8429599432532096,
   0.9251771844789913,
   0.8655339692155823,
   0.9036452083899731,
   0.8206394616536681,
   0.8706483735804971,
   0.8853924473642432,
   0.33799874888632914,
   0.5079750812968089,
   0.8429599432532096,
   0.0},
  raft::distance::DistanceType::Linf,
  0.0};

const InputConfiguration<int, float> input_l1 = {4,
                                                 {0, 1, 1, 2, 4},
                                                 {3, 2, 0, 1},  // indices
                                                 {0.99296, 0.42180, 0.11687, 0.305869},
                                                 {
                                                   // dense output
                                                   0.0,
                                                   0.99296,
                                                   1.41476,
                                                   1.415707,
                                                   0.99296,
                                                   0.0,
                                                   0.42180,
                                                   0.42274,
                                                   1.41476,
                                                   0.42180,
                                                   0.0,
                                                   0.84454,
                                                   1.41570,
                                                   0.42274,
                                                   0.84454,
                                                   0.0,
                                                 },
                                                 raft::distance::DistanceType::L1,
                                                 0.0};

// test dense smem strategy
const std::vector<SparseDistanceCOOSPMVInputs<int, float, dense_smem_strategy_t>>
  inputs_dense_strategy = {{input_inner_product},
                           {input_l2_unexpanded},
                           {input_canberra},
                           {input_lp_unexpanded},
                           {input_linf},
                           {input_l1}};

typedef SparseDistanceCOOSPMVTest<int, float, dense_smem_strategy_t>
  SparseDistanceCOOSPMVTestDenseStrategyF;
TEST_P(SparseDistanceCOOSPMVTestDenseStrategyF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseDistanceCOOSPMVTests,
                        SparseDistanceCOOSPMVTestDenseStrategyF,
                        ::testing::ValuesIn(inputs_dense_strategy));

// test hash and chunk strategy
const std::vector<SparseDistanceCOOSPMVInputs<int, float, hash_strategy_t>> inputs_hash_strategy = {
  {input_inner_product},
  {input_inner_product, 0.5, 2},
  {input_l2_unexpanded},
  {input_l2_unexpanded, 0.5, 2},
  {input_canberra},
  {input_canberra, 0.5, 2},
  {input_canberra, 0.5, 6},
  {input_lp_unexpanded},
  {input_lp_unexpanded, 0.5, 2},
  {input_lp_unexpanded, 0.5, 6},
  {input_linf},
  {input_linf, 0.5, 2},
  {input_linf, 0.5, 6},
  {input_l1},
  {input_l1, 0.5, 2}};

typedef SparseDistanceCOOSPMVTest<int, float, hash_strategy_t>
  SparseDistanceCOOSPMVTestHashStrategyF;
TEST_P(SparseDistanceCOOSPMVTestHashStrategyF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(SparseDistanceCOOSPMVTests,
                        SparseDistanceCOOSPMVTestHashStrategyF,
                        ::testing::ValuesIn(inputs_hash_strategy));

};  // namespace distance
};  // end namespace sparse
};  // end namespace raft
