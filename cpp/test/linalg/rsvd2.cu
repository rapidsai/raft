/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/linalg/rsvd.cuh>
#include <raft/matrix/matrix.cuh>

namespace raft {
namespace linalg {

template <typename T>
struct rsvdInputs {
    T tolerance;
    int len;
    int n_row;
    int n_col;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const rsvdInputs<T>& dims)
{
    return os;
}

template <typename T>
class rsvdTest : public ::testing::TestWithParam<rsvdInputs<T>> {
public:
rsvdTest()
    : params(::testing::TestWithParam<rsvdInputs<T>>::GetParam()),
    stream(handle.get_stream()),
    data(params.len, stream),
    left_eig_vectors_qr(params.n_row * params.n_col, stream),
    right_eig_vectors_trans_qr(params.n_col * params.n_col, stream),
    sing_vals_qr(params.n_col, stream),
    left_eig_vectors_ref(params.n_row * params.n_col, stream),
    right_eig_vectors_ref(params.n_col * params.n_col, stream),
    sing_vals_ref(params.len, stream)
{
}

protected:
void SetUp() override
{
    int len = params.len;

    ASSERT(params.n_row == 3, "This test only supports nrows=3!");
    ASSERT(params.len == 6, "This test only supports len=6!");
    T data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    raft::update_device(data.data(), data_h, len, stream);

    int left_evl  = params.n_row * params.n_col;
    int right_evl = params.n_col * params.n_col;

    T left_eig_vectors_ref_h[] = {-0.308219, -0.906133, -0.289695, 0.488195, 0.110706, -0.865685};

    T right_eig_vectors_ref_h[] = {-0.638636, -0.769509, -0.769509, 0.638636};

    T sing_vals_ref_h[] = {7.065283, 1.040081};

    raft::update_device(left_eig_vectors_ref.data(), left_eig_vectors_ref_h, left_evl, stream);
    raft::update_device(right_eig_vectors_ref.data(), right_eig_vectors_ref_h, right_evl, stream);
    raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, params.n_col, stream);

    randomized_svd(handle,
        raft::make_device_matrix_view(data.data(), params.n_row, params.n_col),
        2,
        4,
        2,
        sing_vals_qr.data(),
        left_eig_vectors_qr.data(),
        right_eig_vectors_trans_qr.data(),
        true,
        true,
        true);
    handle.sync_stream(stream);
}

protected:
raft::handle_t handle;
cudaStream_t stream;

rsvdInputs<T> params;
rmm::device_uvector<T> data, left_eig_vectors_qr, right_eig_vectors_trans_qr, sing_vals_qr,
    left_eig_vectors_ref, right_eig_vectors_ref, sing_vals_ref;
};

const std::vector<rsvdInputs<float>> inputsf1 = {{0.00001f, 3 * 2, 3, 2, 1234ULL}};
const std::vector<rsvdInputs<double>> inputsd1 = {{0.00001, 3 * 2, 3, 2, 1234ULL}};
const std::vector<rsvdInputs<float>> inputsf2 = {{0.00001f, 10 * 8, 10, 8, 1234ULL}};
const std::vector<rsvdInputs<double>> inputsd2 = {{0.00001, 10 * 8, 10, 8, 1234ULL}};

typedef rsvdTest<float> rsvdTestF;
TEST_P(rsvdTestF, Result)
{
    ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
    ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
    /*ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_trans_qr.data(),
                                params.n_col * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));*/
}

typedef rsvdTest<double> rsvdTestD;
TEST_P(rsvdTestD, Result)
{
    ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
    ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
    /*ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_trans_qr.data(),
                                params.n_col * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));*/
}

INSTANTIATE_TEST_SUITE_P(rsvdTests, rsvdTestF, ::testing::ValuesIn(inputsf1));
INSTANTIATE_TEST_SUITE_P(rsvdTests, rsvdTestD, ::testing::ValuesIn(inputsd1));
//INSTANTIATE_TEST_SUITE_P(rsvdTests, rsvdTestF, ::testing::ValuesIn(inputsf2));
//INSTANTIATE_TEST_SUITE_P(rsvdTests, rsvdTestD, ::testing::ValuesIn(inputsd2));
}  // end namespace linalg
}  // end namespace raft
