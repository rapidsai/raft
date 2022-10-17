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
#include <raft/linalg/svd.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/random/rng.cuh>

namespace raft {
namespace linalg {

template <typename T>
struct randomized_svdInputs {
    T tolerance;
    int n_row;
    int n_col;
    int k;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const randomized_svdInputs<T>& dims)
{
    return os;
}

template <typename T>
class randomized_svdTest : public ::testing::TestWithParam<randomized_svdInputs<T>> {
public:
randomized_svdTest()
    : params(::testing::TestWithParam<randomized_svdInputs<T>>::GetParam()),
    stream(handle.get_stream()),
    data(params.n_row * params.n_col, stream),
    left_eig_vectors_act(params.n_row * params.n_row, stream),
    right_eig_vectors_trans_act(params.n_col * params.n_col, stream),
    sing_vals_act(std::min(params.n_row, params.n_col), stream),
    left_eig_vectors_ref(params.n_row * params.n_row, stream),
    right_eig_vectors_ref(params.n_col * params.n_col, stream),
    sing_vals_ref(std::min(params.n_row, params.n_col), stream)
{
}

protected:
void basicTest()
{
    int len = params.n_row * params.n_col;
    ASSERT(params.n_row == 4 && params.n_col == 4, "This test only supports nrows=4 && ncols=4!");
    T data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 4.0, 2.0, 1.0, 3.0, 1.0, 1.0, 0.0, -1.0, 2.0, 0.0};
    raft::update_device(data.data(), data_h, len, stream);

    int left_evl  = params.n_row * params.n_row;
    int right_evl = params.n_col * params.n_col;

    T left_eig_vectors_ref_h[] = {0.57627357, -0.43264557, 0.46833317,  0.83332313, 0.56581465, -0.32344445, 0.35836657,  0.11736295};

    T right_eig_vectors_ref_h[] = {0.52344558,  0.77088736,  0.35385256,  0.08078222, 0.59393026, -0.57017593,  0.44422555, -0.35328447};

    T sing_vals_ref_h[] = {8.21091783, 4.18985879};

    raft::update_device(left_eig_vectors_ref.data(), left_eig_vectors_ref_h, left_evl, stream);
    raft::update_device(right_eig_vectors_ref.data(), right_eig_vectors_ref_h, right_evl, stream);
    raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, std::min(params.n_row, params.n_col), stream);

    randomized_svd(handle,
        raft::make_device_matrix_view(data.data(), params.n_row, params.n_col),
        raft::make_device_vector_view(sing_vals_act.data(), std::min(params.n_row, params.n_col)),
        raft::make_device_matrix_view(left_eig_vectors_act.data(), params.n_row, params.n_row),
        raft::make_device_matrix_view(right_eig_vectors_trans_act.data(), params.n_col, params.n_col),
        params.k,
        1,
        2,
        true,
        true,
        true);
    handle.sync_stream(stream);
}

void advancedTest()
{
    int len = params.n_row * params.n_col;
    int left_evl  = params.n_row * params.n_row;
    int right_evl = params.n_col * params.n_col;
    raft::random::RngState r(params.seed);
    uniform(handle, r, data.data(), len, T(-1.0), T(2.0));
    
    svd_qr(handle,
        raft::make_device_matrix_view<const T, uint32_t, raft::col_major>(data.data(), params.n_row, params.n_col),
        raft::make_device_vector_view<T, uint32_t>(sing_vals_ref.data(), params.k));
    handle.sync_stream(stream);

    randomized_svd(handle,
        raft::make_device_matrix_view<T, uint32_t, raft::col_major>(data.data(), params.n_row, params.n_col),
        raft::make_device_vector_view<T, uint32_t>(sing_vals_act.data(), std::min(params.n_row, params.n_col)),
        raft::make_device_matrix_view<T, uint32_t, raft::col_major>(left_eig_vectors_act.data(), params.n_row, params.n_row),
        raft::make_device_matrix_view<T, uint32_t, raft::col_major>(right_eig_vectors_trans_act.data(), params.n_col, params.n_col),
        params.k,
        2 * params.k,
        2,
        true,
        true,
        true);
    handle.sync_stream(stream);
}

void SetUp() override
{
    if (params.n_row == 4 && params.n_col == 4)
    {
        basicTest();
    } else 
    {
        advancedTest();
    }
}

protected:
raft::handle_t handle;
cudaStream_t stream;

randomized_svdInputs<T> params;
rmm::device_uvector<T> data, left_eig_vectors_act, right_eig_vectors_trans_act, sing_vals_act,
    left_eig_vectors_ref, right_eig_vectors_ref, sing_vals_ref;
};

const std::vector<randomized_svdInputs<float>> inputsf1 = {{0.00001f, 4, 4, 2, 1234ULL}};
const std::vector<randomized_svdInputs<double>> inputsd1 = {{0.00001, 4, 4, 2, 1234ULL}};
const std::vector<randomized_svdInputs<float>> inputsf2 = {{0.00001f, 300, 80, 15, 1234ULL}};
const std::vector<randomized_svdInputs<double>> inputsd2 = {{0.00001, 300, 80, 15, 1234ULL}};

typedef randomized_svdTest<float> randomized_svdTestF;
TEST_P(randomized_svdTestF, Result)
{
    ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_act.data(),
                                params.k,
                                raft::CompareApproxAbs<float>(params.tolerance)));
    if (params.k > 10) {
        ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                    left_eig_vectors_act.data(),
                                    params.n_row * params.n_row,
                                    raft::CompareApproxAbs<float>(params.tolerance)));
        ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                    right_eig_vectors_trans_act.data(),
                                    params.n_col * params.n_col,
                                    raft::CompareApproxAbs<float>(params.tolerance)));
    }
}

typedef randomized_svdTest<double> randomized_svdTestD;
TEST_P(randomized_svdTestD, Result)
{
    ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_act.data(),
                                params.k,
                                raft::CompareApproxAbs<double>(params.tolerance)));
    if (params.k > 10) {
        ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                    left_eig_vectors_act.data(),
                                    params.n_row * params.n_row,
                                    raft::CompareApproxAbs<double>(params.tolerance)));
        ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                    right_eig_vectors_trans_act.data(),
                                    params.n_col * params.n_col,
                                    raft::CompareApproxAbs<double>(params.tolerance)));
    }
}

INSTANTIATE_TEST_SUITE_P(randomized_svdTests1, randomized_svdTestF, ::testing::ValuesIn(inputsf1));
INSTANTIATE_TEST_SUITE_P(randomized_svdTests1, randomized_svdTestD, ::testing::ValuesIn(inputsd1));
INSTANTIATE_TEST_SUITE_P(randomized_svdTests2, randomized_svdTestF, ::testing::ValuesIn(inputsf2));
INSTANTIATE_TEST_SUITE_P(randomized_svdTests2, randomized_svdTestD, ::testing::ValuesIn(inputsd2));
}  // end namespace linalg
}  // end namespace raft
