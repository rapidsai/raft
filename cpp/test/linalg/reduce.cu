/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include "reduce.cuh"
#include <gtest/gtest.h>
#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename IdxType>
struct ReduceInputs {
  OutType tolerance;
  IdxType rows, cols;
  bool rowMajor, alongRows;
  OutType init;
  unsigned long long int seed;
};

template <typename InType, typename OutType, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const ReduceInputs<InType, OutType, IdxType>& dims)
{
  os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", " << dims.rowMajor
     << ", " << dims.alongRows << ", " << dims.init << " " << dims.seed << '}';
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
void reduceLaunch(OutType* dots,
                  const InType* data,
                  IdxType cols,
                  IdxType rows,
                  bool rowMajor,
                  bool alongRows,
                  OutType init,
                  bool inplace,
                  cudaStream_t stream,
                  MainLambda main_op,
                  ReduceLambda reduce_op,
                  FinalLambda final_op)
{
  Apply apply         = alongRows ? Apply::ALONG_ROWS : Apply::ALONG_COLUMNS;
  IdxType output_size = alongRows ? cols : rows;

  auto output_view          = raft::make_device_vector_view(dots, output_size);
  auto input_view_row_major = raft::make_device_matrix_view(data, rows, cols);
  auto input_view_col_major =
    raft::make_device_matrix_view<const InType, IdxType, raft::col_major>(data, rows, cols);

  raft::resources handle;
  resource::set_cuda_stream(handle, stream);

  if (rowMajor) {
    reduce(handle,
           input_view_row_major,
           output_view,
           init,
           apply,
           inplace,
           main_op,
           reduce_op,
           final_op);
  } else {
    reduce(handle,
           input_view_col_major,
           output_view,
           init,
           apply,
           inplace,
           main_op,
           reduce_op,
           final_op);
  }
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda   = raft::sq_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::sqrt_op>
class ReduceTest : public ::testing::TestWithParam<ReduceInputs<InType, OutType, IdxType>> {
 public:
  ReduceTest()
    : params(::testing::TestWithParam<ReduceInputs<InType, OutType, IdxType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      dots_exp(params.alongRows ? params.rows : params.cols, stream),
      dots_act(params.alongRows ? params.rows : params.cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    IdxType rows = params.rows, cols = params.cols;
    IdxType len = rows * cols;
    gen_uniform(handle, data.data(), r, len);

    MainLambda main_op;
    ReduceLambda reduce_op;
    FinalLambda fin_op;

    // For both the naive and the actual implementation, execute first with inplace=false then true

    naiveReduction(dots_exp.data(),
                   data.data(),
                   cols,
                   rows,
                   params.rowMajor,
                   params.alongRows,
                   stream,
                   params.init,
                   false,
                   main_op,
                   reduce_op,
                   fin_op);
    naiveReduction(dots_exp.data(),
                   data.data(),
                   cols,
                   rows,
                   params.rowMajor,
                   params.alongRows,
                   stream,
                   params.init,
                   true,
                   main_op,
                   reduce_op,
                   fin_op);

    reduceLaunch(dots_act.data(),
                 data.data(),
                 cols,
                 rows,
                 params.rowMajor,
                 params.alongRows,
                 params.init,
                 false,
                 stream,
                 main_op,
                 reduce_op,
                 fin_op);
    reduceLaunch(dots_act.data(),
                 data.data(),
                 cols,
                 rows,
                 params.rowMajor,
                 params.alongRows,
                 params.init,
                 true,
                 stream,
                 main_op,
                 reduce_op,
                 fin_op);

    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  ReduceInputs<InType, OutType, IdxType> params;
  rmm::device_uvector<InType> data;
  rmm::device_uvector<OutType> dots_exp, dots_act;
};

#define REDUCE_TEST(test_type, test_name, test_inputs)                                            \
  typedef RAFT_DEPAREN(test_type) test_name;                                                      \
  TEST_P(test_name, Result)                                                                       \
  {                                                                                               \
    ASSERT_TRUE(raft::devArrMatch(                                                                \
      dots_exp.data(), dots_act.data(), dots_exp.size(), raft::CompareApprox(params.tolerance))); \
  }                                                                                               \
  INSTANTIATE_TEST_CASE_P(ReduceTests, test_name, ::testing::ValuesIn(test_inputs))

const std::vector<ReduceInputs<float, float, int>> inputsff_i32 =
  raft::util::itertools::product<ReduceInputs<float, float, int>>(
    {0.000002f}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputsdd_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000000001}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0}, {1234ULL});
const std::vector<ReduceInputs<float, double, int>> inputsfd_i32 =
  raft::util::itertools::product<ReduceInputs<float, double, int>>(
    {0.000000001}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputsff_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>({0.000002f},
                                                                       {11u, 1234u},
                                                                       {7u, 33u, 128u, 500u},
                                                                       {true, false},
                                                                       {true, false},
                                                                       {0.0f},
                                                                       {1234ULL});
const std::vector<ReduceInputs<float, float, int64_t>> inputsff_i64 =
  raft::util::itertools::product<ReduceInputs<float, float, int64_t>>(
    {0.000002f}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0f}, {1234ULL});

REDUCE_TEST((ReduceTest<float, float, int>), ReduceTestFFI32, inputsff_i32);
REDUCE_TEST((ReduceTest<double, double, int>), ReduceTestDDI32, inputsdd_i32);
REDUCE_TEST((ReduceTest<float, double, int>), ReduceTestFDI32, inputsfd_i32);
REDUCE_TEST((ReduceTest<float, float, uint32_t>), ReduceTestFFU32, inputsff_u32);
REDUCE_TEST((ReduceTest<float, float, int64_t>), ReduceTestFFI64, inputsff_i64);

// The following test cases are for "thick" coalesced reductions

const std::vector<ReduceInputs<float, float, int>> inputsff_thick_i32 =
  raft::util::itertools::product<ReduceInputs<float, float, int>>(
    {0.0001f}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputsdd_thick_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000001}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {0.0}, {1234ULL});
const std::vector<ReduceInputs<float, double, int>> inputsfd_thick_i32 =
  raft::util::itertools::product<ReduceInputs<float, double, int>>(
    {0.000001}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputsff_thick_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>(
    {0.0001f}, {3u, 9u}, {17771u, 33333u, 100000u}, {true}, {true}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<float, float, int64_t>> inputsff_thick_i64 =
  raft::util::itertools::product<ReduceInputs<float, float, int64_t>>(
    {0.0001f}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {0.0f}, {1234ULL});

REDUCE_TEST((ReduceTest<float, float, int>), ReduceTestFFI32Thick, inputsff_thick_i32);
REDUCE_TEST((ReduceTest<double, double, int>), ReduceTestDDI32Thick, inputsdd_thick_i32);
REDUCE_TEST((ReduceTest<float, double, int>), ReduceTestFDI32Thick, inputsfd_thick_i32);
REDUCE_TEST((ReduceTest<float, float, uint32_t>), ReduceTestFFU32Thick, inputsff_thick_u32);
REDUCE_TEST((ReduceTest<float, float, int64_t>), ReduceTestFFI64Thick, inputsff_thick_i64);

// Test key-value-pair reductions. This is important because shuffle intrinsics can't be used
// directly with those types.

template <typename T, typename IdxT = int>
struct ValueToKVP {
  HDI raft::KeyValuePair<IdxT, T> operator()(T value, IdxT idx) { return {idx, value}; }
};

template <typename T1, typename T2>
struct ArgMaxOp {
  HDI raft::KeyValuePair<T1, T2> operator()(raft::KeyValuePair<T1, T2> a,
                                            raft::KeyValuePair<T1, T2> b)
  {
    return (a.value > b.value || (a.value == b.value && a.key <= b.key)) ? a : b;
  }
};

const std::vector<ReduceInputs<short, raft::KeyValuePair<int, short>, int>> inputs_kvpis_i32 =
  raft::util::itertools::product<ReduceInputs<short, raft::KeyValuePair<int, short>, int>>(
    {raft::KeyValuePair{0, short(0)}},
    {11, 1234},
    {7, 33, 128, 500},
    {true},
    {true},
    {raft::KeyValuePair{0, short(0)}},
    {1234ULL});
const std::vector<ReduceInputs<float, raft::KeyValuePair<int, float>, int>> inputs_kvpif_i32 =
  raft::util::itertools::product<ReduceInputs<float, raft::KeyValuePair<int, float>, int>>(
    {raft::KeyValuePair{0, 0.0001f}},
    {11, 1234},
    {7, 33, 128, 500},
    {true},
    {true},
    {raft::KeyValuePair{0, 0.0f}},
    {1234ULL});
const std::vector<ReduceInputs<double, raft::KeyValuePair<int, double>, int>> inputs_kvpid_i32 =
  raft::util::itertools::product<ReduceInputs<double, raft::KeyValuePair<int, double>, int>>(
    {raft::KeyValuePair{0, 0.000001}},
    {11, 1234},
    {7, 33, 128, 500},
    {true},
    {true},
    {raft::KeyValuePair{0, 0.0}},
    {1234ULL});

REDUCE_TEST((ReduceTest<short,
                        raft::KeyValuePair<int, short>,
                        int,
                        ValueToKVP<short, int>,
                        ArgMaxOp<int, short>,
                        raft::identity_op>),
            ReduceTestKVPISI32,
            inputs_kvpis_i32);
REDUCE_TEST((ReduceTest<float,
                        raft::KeyValuePair<int, float>,
                        int,
                        ValueToKVP<float, int>,
                        ArgMaxOp<int, float>,
                        raft::identity_op>),
            ReduceTestKVPIFI32,
            inputs_kvpif_i32);
REDUCE_TEST((ReduceTest<double,
                        raft::KeyValuePair<int, double>,
                        int,
                        ValueToKVP<double, int>,
                        ArgMaxOp<int, double>,
                        raft::identity_op>),
            ReduceTestKVPIDI32,
            inputs_kvpid_i32);

}  // end namespace linalg
}  // end namespace raft
