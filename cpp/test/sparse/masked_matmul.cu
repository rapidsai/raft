/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/linalg/masked_matmul.hpp>
#include <raft/util/cudart_utils.hpp>

#include <thrust/reduce.h>

#include <cusparse.h>
#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename value_t, typename output_t, typename index_t>
struct MaskedMatmulInputs {
  output_t tolerance;

  index_t m;
  index_t k;
  index_t n;

  float sparsity;

  unsigned long long int seed;
};

template <typename value_t>
struct sum_abs_op {
  __host__ __device__ value_t operator()(const value_t& x, const value_t& y) const
  {
    return y >= value_t(0.0) ? (x + y) : (x - y);
  }
};

struct float_to_half {
  __host__ __device__ __half operator()(const float x) const { return __float2half(x); }
};

template <typename value_t, typename output_t, typename index_t>
::std::ostream& operator<<(::std::ostream& os,
                           const MaskedMatmulInputs<value_t, output_t, index_t>& params)
{
  os << " m: " << params.m << "\tk: " << params.k << "\tn: " << params.n
     << "\tsparsity: " << params.sparsity;

  return os;
}

bool isCuSparseVersionGreaterThan_12_0_1()
{
  int version;
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseGetVersion(handle, &version);

  int major = version / 1000;
  int minor = (version % 1000) / 100;
  int patch = version % 100;

  cusparseDestroy(handle);

  return (major > 12) || (major == 12 && minor > 0) || (major == 12 && minor == 0 && patch >= 2);
}

template <typename value_t,
          typename output_t,
          typename index_t,
          typename bitmap_t      = uint32_t,
          typename LayoutPolicyA = raft::row_major,
          typename LayoutPolicyB = raft::row_major>
class MaskedMatmulTest
  : public ::testing::TestWithParam<MaskedMatmulInputs<value_t, output_t, index_t>> {
 public:
  MaskedMatmulTest()
    : params(::testing::TestWithParam<MaskedMatmulInputs<value_t, output_t, index_t>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      a_data_d(0, resource::get_cuda_stream(handle)),
      b_data_d(0, resource::get_cuda_stream(handle)),
      bitmap_d(0, resource::get_cuda_stream(handle)),
      c_indptr_d(0, resource::get_cuda_stream(handle)),
      c_indices_d(0, resource::get_cuda_stream(handle)),
      c_data_d(0, resource::get_cuda_stream(handle)),
      c_expected_data_d(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  index_t create_sparse_matrix(index_t m, index_t n, float sparsity, std::vector<bitmap_t>& bitmap)
  {
    index_t total    = static_cast<index_t>(m * n);
    index_t num_ones = static_cast<index_t>((total * 1.0f) * sparsity);
    index_t res      = num_ones;

    for (auto& item : bitmap) {
      item = static_cast<bitmap_t>(0);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<index_t> dis(0, total - 1);

    while (num_ones > 0) {
      index_t index = dis(gen);

      bitmap_t& element    = bitmap[index / (8 * sizeof(bitmap_t))];
      index_t bit_position = index % (8 * sizeof(bitmap_t));

      if (((element >> bit_position) & 1) == 0) {
        element |= (static_cast<index_t>(1) << bit_position);
        num_ones--;
      }
    }
    return res;
  }

  void cpu_convert_to_csr(std::vector<bitmap_t>& bitmap,
                          index_t rows,
                          index_t cols,
                          std::vector<index_t>& indices,
                          std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    index_t index        = 0;
    bitmap_t element     = 0;
    index_t bit_position = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        index        = i * cols + j;
        element      = bitmap[index / (8 * sizeof(bitmap_t))];
        bit_position = index % (8 * sizeof(bitmap_t));

        if (((element >> bit_position) & 1)) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  void cpu_sddmm(const std::vector<value_t>& A,
                 const std::vector<value_t>& B,
                 std::vector<output_t>& vals,
                 const std::vector<index_t>& cols,
                 const std::vector<index_t>& row_ptrs,
                 bool is_row_major_A,
                 bool is_row_major_B)
  {
    if (params.m * params.k != static_cast<index_t>(A.size()) ||
        params.k * params.n != static_cast<index_t>(B.size())) {
      std::cerr << "Matrix dimensions and vector size do not match!" << std::endl;
      return;
    }

    for (index_t i = 0; i < params.m; ++i) {
      for (index_t j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
        output_t sum = 0;
        for (index_t l = 0; l < params.k; ++l) {
          index_t a_index = i * params.k + l;
          index_t b_index = cols[j] * params.k + l;
          if constexpr ((std::is_same_v<output_t, float> && std::is_same_v<value_t, half>)) {
            sum += __half2float(A[a_index]) * __half2float(B[b_index]);
          } else {
            sum += A[a_index] * B[b_index];
          }
        }
        vals[j] = sum;
      }
    }
  }

  void make_data()
  {
    index_t a_size = params.m * params.k;
    index_t b_size = params.k * params.n;
    index_t c_size = params.m * params.n;

    index_t element = raft::ceildiv(params.m * params.n, index_t(sizeof(bitmap_t) * 8));
    std::vector<bitmap_t> bitmap_h(element);

    std::vector<value_t> a_data_h(a_size);
    std::vector<value_t> b_data_h(b_size);

    a_data_d.resize(a_size, stream);
    b_data_d.resize(b_size, stream);
    bitmap_d.resize(bitmap_h.size(), stream);

    auto blobs_a_b = raft::make_device_matrix<output_t, index_t>(handle, 1, a_size + b_size);
    auto labels    = raft::make_device_vector<index_t, index_t>(handle, 1);

    raft::random::make_blobs<output_t, index_t>(blobs_a_b.data_handle(),
                                                labels.data_handle(),
                                                1,
                                                a_size + b_size,
                                                1,
                                                stream,
                                                false,
                                                nullptr,
                                                nullptr,
                                                output_t(1.0),
                                                false,
                                                output_t(-1.0f),
                                                output_t(1.0f),
                                                uint64_t(2024));

    if constexpr ((std::is_same_v<output_t, float> && std::is_same_v<value_t, half>)) {
      {
        thrust::device_ptr<output_t> d_output_ptr =
          thrust::device_pointer_cast(blobs_a_b.data_handle());
        thrust::device_ptr<value_t> d_value_ptr = thrust::device_pointer_cast(a_data_d.data());
        thrust::transform(thrust::cuda::par.on(stream),
                          d_output_ptr,
                          d_output_ptr + a_size,
                          d_value_ptr,
                          float_to_half());
      }
      {
        thrust::device_ptr<output_t> d_output_ptr =
          thrust::device_pointer_cast(blobs_a_b.data_handle() + a_size);
        thrust::device_ptr<value_t> d_value_ptr = thrust::device_pointer_cast(b_data_d.data());
        thrust::transform(thrust::cuda::par.on(stream),
                          d_output_ptr,
                          d_output_ptr + b_size,
                          d_value_ptr,
                          float_to_half());
      }
      raft::copy(a_data_h.data(), a_data_d.data(), a_size, stream);
      raft::copy(b_data_h.data(), b_data_d.data(), b_size, stream);
    } else {
      raft::copy(a_data_h.data(), blobs_a_b.data_handle(), a_size, stream);
      raft::copy(b_data_h.data(), blobs_a_b.data_handle() + a_size, b_size, stream);

      raft::copy(a_data_d.data(), blobs_a_b.data_handle(), a_size, stream);
      raft::copy(b_data_d.data(), blobs_a_b.data_handle() + a_size, b_size, stream);
    }

    resource::sync_stream(handle);

    index_t c_true_nnz = create_sparse_matrix(params.m, params.n, params.sparsity, bitmap_h);

    std::vector<index_t> c_indptr_h(params.m + 1);
    std::vector<index_t> c_indices_h(c_true_nnz);
    std::vector<output_t> c_data_h(c_true_nnz);

    cpu_convert_to_csr(bitmap_h, params.m, params.n, c_indices_h, c_indptr_h);

    c_data_d.resize(c_data_h.size(), stream);

    update_device(c_data_d.data(), c_data_h.data(), c_data_h.size(), stream);
    update_device(bitmap_d.data(), bitmap_h.data(), bitmap_h.size(), stream);
    resource::sync_stream(handle);

    cpu_sddmm(a_data_h, b_data_h, c_data_h, c_indices_h, c_indptr_h, true, true);

    c_indptr_d.resize(c_indptr_h.size(), stream);
    c_indices_d.resize(c_indices_h.size(), stream);
    c_expected_data_d.resize(c_data_h.size(), stream);

    update_device(c_indptr_d.data(), c_indptr_h.data(), c_indptr_h.size(), stream);
    update_device(c_indices_d.data(), c_indices_h.data(), c_indices_h.size(), stream);
    update_device(c_expected_data_d.data(), c_data_h.data(), c_data_h.size(), stream);

    resource::sync_stream(handle);
  }

  void SetUp() override
  {
    if (std::is_same_v<value_t, half> && !isCuSparseVersionGreaterThan_12_0_1()) {
      GTEST_SKIP() << "Skipping all tests for half-float as cuSparse doesn't support it.";
    }
    make_data();
  }

  void Run()
  {
    auto A =
      raft::make_device_matrix_view<const value_t, index_t>(a_data_d.data(), params.m, params.k);
    auto B =
      raft::make_device_matrix_view<const value_t, index_t>(b_data_d.data(), params.n, params.k);

    auto mask =
      raft::core::bitmap_view<const bitmap_t, index_t>(bitmap_d.data(), params.m, params.n);

    auto c_structure = raft::make_device_compressed_structure_view<index_t, index_t, index_t>(
      c_indptr_d.data(),
      c_indices_d.data(),
      params.m,
      params.n,
      static_cast<index_t>(c_indices_d.size()));

    auto C = raft::make_device_csr_matrix_view<output_t>(c_data_d.data(), c_structure);

    raft::sparse::linalg::masked_matmul(handle, A, B, mask, C);

    resource::sync_stream(handle);

    ASSERT_TRUE(raft::devArrMatch<output_t>(c_expected_data_d.data(),
                                            C.get_elements().data(),
                                            c_expected_data_d.size(),
                                            raft::CompareApprox<output_t>(params.tolerance),
                                            stream));

    thrust::device_ptr<output_t> expected_data_ptr =
      thrust::device_pointer_cast(c_expected_data_d.data());
    output_t sum_abs = thrust::reduce(thrust::cuda::par.on(stream),
                                      expected_data_ptr,
                                      expected_data_ptr + c_expected_data_d.size(),
                                      output_t(0.0f),
                                      sum_abs_op<output_t>());
    output_t avg     = sum_abs / (1.0f * c_expected_data_d.size());

    ASSERT_GE(avg, (params.tolerance * static_cast<output_t>(0.001f)));
  }

  raft::resources handle;
  cudaStream_t stream;
  MaskedMatmulInputs<value_t, output_t, index_t> params;

  rmm::device_uvector<value_t> a_data_d;
  rmm::device_uvector<value_t> b_data_d;
  rmm::device_uvector<bitmap_t> bitmap_d;

  rmm::device_uvector<index_t> c_indptr_d;
  rmm::device_uvector<index_t> c_indices_d;
  rmm::device_uvector<output_t> c_data_d;

  rmm::device_uvector<output_t> c_expected_data_d;
};

using MaskedMatmulTestF = MaskedMatmulTest<float, float, int>;
TEST_P(MaskedMatmulTestF, Result) { Run(); }

using MaskedMatmulTestD = MaskedMatmulTest<double, double, int>;
TEST_P(MaskedMatmulTestD, Result) { Run(); }

using MaskedMatmulTestH = MaskedMatmulTest<half, float, int>;
TEST_P(MaskedMatmulTestH, Result) { Run(); }

const std::vector<MaskedMatmulInputs<float, float, int>> sddmm_inputs_f = {
  {0.001f, 2, 255, 1023, 0.19, 1234ULL},
  {0.001f, 2, 255, 1023 * 2, 0.19, 1234ULL},
  {0.001f, 2, 255, 1023 * 3, 0.38, 1234ULL},
  {0.0001f, 10, 255, 13000, 0.01, 1234ULL},
  {0.0001f, 10, 5, 32, 0.1, 1234ULL},
  {0.001f, 11, 255, 1023, 0.19, 1234ULL},
  {0.001f, 11, 255, 1023 * 2, 0.19, 1234ULL},
  {0.001f, 11, 255, 1023 * 3, 0.38, 1234ULL},
  {0.0003f, 32, 1024, 1024, 0.2, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.1, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.19, 1234ULL},
  {0.001f, 1023, 1023, 1023 * 3, 0.38, 1234ULL},
  {0.001f, 1025, 1025, 1025 * 3, 0.31, 1234ULL},
  {0.0001f, 1024, 1024, 32, 0.3, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.4, 1234ULL},
  {0.0003f, 31, 1025, 1025, 0.19, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.1, 1234ULL}};

const std::vector<MaskedMatmulInputs<double, double, int>> sddmm_inputs_d = {
  {0.0001f, 2, 255, 1023, 0.19, 1234ULL},
  {0.0001f, 2, 255, 1023 * 2, 0.19, 1234ULL},
  {0.0001f, 2, 255, 1023 * 3, 0.38, 1234ULL},
  {0.0001f, 10, 255, 13000, 0.01, 1234ULL},
  {0.0001f, 10, 5, 32, 0.1, 1234ULL},
  {0.0001f, 11, 255, 1023, 0.19, 1234ULL},
  {0.0001f, 11, 255, 1023 * 2, 0.19, 1234ULL},
  {0.0001f, 11, 255, 1023 * 3, 0.38, 1234ULL},
  {0.0001f, 32, 1024, 1024, 0.2, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.1, 1234ULL},
  {0.0001f, 1024, 1024, 1024, 0.19, 1234ULL},
  {0.0001f, 1023, 1023, 1023 * 3, 0.38, 1234ULL},
  {0.0001f, 1025, 1025, 1025 * 3, 0.31, 1234ULL},
  {0.0001f, 1024, 1024, 32, 0.3, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.4, 1234ULL},
  {0.0001f, 31, 1025, 1025, 0.19, 1234ULL},
  {0.0001f, 1024, 1024, 1024, 0.1, 1234ULL}};

const std::vector<MaskedMatmulInputs<half, float, int>> sddmm_inputs_h = {
  {0.001f, 2, 255, 1023, 0.19, 1234ULL},
  {0.001f, 2, 255, 1023 * 2, 0.19, 1234ULL},
  {0.001f, 2, 255, 1023 * 3, 0.38, 1234ULL},
  {0.0001f, 10, 255, 13000, 0.01, 1234ULL},
  {0.0001f, 10, 5, 32, 0.1, 1234ULL},
  {0.001f, 11, 255, 1023, 0.19, 1234ULL},
  {0.001f, 11, 255, 1023 * 2, 0.19, 1234ULL},
  {0.001f, 11, 255, 1023 * 3, 0.38, 1234ULL},
  {0.0003f, 32, 1024, 1024, 0.2, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.1, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.19, 1234ULL},
  {0.001f, 1023, 1023, 1023 * 3, 0.38, 1234ULL},
  {0.001f, 1025, 1025, 1025 * 3, 0.31, 1234ULL},
  {0.0001f, 1024, 1024, 32, 0.3, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.4, 1234ULL},
  {0.0003f, 31, 1025, 1025, 0.19, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.1, 1234ULL}};

INSTANTIATE_TEST_CASE_P(MaskedMatmulTest, MaskedMatmulTestF, ::testing::ValuesIn(sddmm_inputs_f));

INSTANTIATE_TEST_CASE_P(MaskedMatmulTest, MaskedMatmulTestD, ::testing::ValuesIn(sddmm_inputs_d));

INSTANTIATE_TEST_CASE_P(MaskedMatmulTest, MaskedMatmulTestH, ::testing::ValuesIn(sddmm_inputs_h));

}  // namespace sparse
}  // namespace raft
