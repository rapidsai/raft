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

#include "../test_utils.h"
#include "reduce.cuh"
#include <gtest/gtest.h>
#include <raft/core/detail/macros.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/random/rng.cuh>
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
  unsigned long long int seed;
};

template <typename InType, typename OutType, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const ReduceInputs<InType, OutType, IdxType>& dims)
{
  os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", " << dims.rowMajor
     << ", " << dims.alongRows << ", " << dims.seed << '}' << std::endl;
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename OutType, typename IdxType>
void reduceLaunch(OutType* dots,
                  const InType* data,
                  IdxType cols,
                  IdxType rows,
                  bool rowMajor,
                  bool alongRows,
                  bool inplace,
                  cudaStream_t stream)
{
  Apply apply         = alongRows ? Apply::ALONG_ROWS : Apply::ALONG_COLUMNS;
  IdxType output_size = alongRows ? cols : rows;

  auto output_view_row_major = raft::make_device_vector_view(dots, output_size);
  auto input_view_row_major  = raft::make_device_matrix_view(data, rows, cols);

  auto output_view_col_major = raft::make_device_vector_view<OutType, IdxType>(dots, output_size);
  auto input_view_col_major =
    raft::make_device_matrix_view<const InType, IdxType, raft::col_major>(data, rows, cols);

  raft::handle_t handle{stream};

  if (rowMajor) {
    reduce(handle,
           input_view_row_major,
           output_view_row_major,
           (OutType)0,
           apply,
           inplace,
           [] __device__(InType in, IdxType i) { return static_cast<OutType>(in * in); });
  } else {
    reduce(handle,
           input_view_col_major,
           output_view_col_major,
           (OutType)0,
           apply,
           inplace,
           [] __device__(InType in, IdxType i) { return static_cast<OutType>(in * in); });
  }
}

template <typename InType, typename OutType, typename IdxType>
class ReduceTest : public ::testing::TestWithParam<ReduceInputs<InType, OutType, IdxType>> {
 public:
  ReduceTest()
    : params(::testing::TestWithParam<ReduceInputs<InType, OutType, IdxType>>::GetParam()),
      stream(handle.get_stream()),
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
    uniform(handle, r, data.data(), len, InType(-1.0), InType(1.0));
    naiveReduction(
      dots_exp.data(), data.data(), cols, rows, params.rowMajor, params.alongRows, stream);

    // Perform reduction with default inplace = false first
    reduceLaunch(
      dots_act.data(), data.data(), cols, rows, params.rowMajor, params.alongRows, false, stream);
    // Add to result with inplace = true next, which shouldn't affect
    // in the case of coalescedReduction!
    if (!(params.rowMajor ^ params.alongRows)) {
      reduceLaunch(
        dots_act.data(), data.data(), cols, rows, params.rowMajor, params.alongRows, true, stream);
    }
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
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
    {0.000002f}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputsdd_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000000001}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {1234ULL});
const std::vector<ReduceInputs<float, double, int>> inputsfd_i32 =
  raft::util::itertools::product<ReduceInputs<float, double, int>>(
    {0.000000001}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputsff_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>(
    {0.000002f}, {11u, 1234u}, {7u, 33u, 128u, 500u}, {true, false}, {true, false}, {1234ULL});
const std::vector<ReduceInputs<float, float, int64_t>> inputsff_i64 =
  raft::util::itertools::product<ReduceInputs<float, float, int64_t>>(
    {0.000002f}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {1234ULL});

REDUCE_TEST((ReduceTest<float, float, int>), ReduceTestFFI32, inputsff_i32);
REDUCE_TEST((ReduceTest<double, double, int>), ReduceTestDDI32, inputsdd_i32);
REDUCE_TEST((ReduceTest<float, double, int>), ReduceTestFDI32, inputsfd_i32);
REDUCE_TEST((ReduceTest<float, float, uint32_t>), ReduceTestFFU32, inputsff_u32);
REDUCE_TEST((ReduceTest<float, float, int64_t>), ReduceTestFFI64, inputsff_i64);

// The following test cases are for "thick" coalesced reductions

const std::vector<ReduceInputs<float, float, int>> inputsff_thick_i32 =
  raft::util::itertools::product<ReduceInputs<float, float, int>>(
    {0.0001f}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputsdd_thick_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000001}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {1234ULL});
const std::vector<ReduceInputs<float, double, int>> inputsfd_thick_i32 =
  raft::util::itertools::product<ReduceInputs<float, double, int>>(
    {0.000001}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputsff_thick_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>(
    {0.0001f}, {3u, 9u}, {17771u, 33333u, 100000u}, {true}, {true}, {1234ULL});
const std::vector<ReduceInputs<float, float, int64_t>> inputsff_thick_i64 =
  raft::util::itertools::product<ReduceInputs<float, float, int64_t>>(
    {0.0001f}, {3, 9}, {17771, 33333, 100000}, {true}, {true}, {1234ULL});

REDUCE_TEST((ReduceTest<float, float, int>), ReduceTestFFI32Thick, inputsff_thick_i32);
REDUCE_TEST((ReduceTest<double, double, int>), ReduceTestDDI32Thick, inputsdd_thick_i32);
REDUCE_TEST((ReduceTest<float, double, int>), ReduceTestFDI32Thick, inputsfd_thick_i32);
REDUCE_TEST((ReduceTest<float, float, uint32_t>), ReduceTestFFU32Thick, inputsff_thick_u32);
REDUCE_TEST((ReduceTest<float, float, int64_t>), ReduceTestFFI64Thick, inputsff_thick_i64);

}  // end namespace linalg
}  // end namespace raft
