/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "unary_op.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

/*
 * Padded_float is a 12 byte type that contains a single float. Two integers are
 * used for padding. It is used to test types that are not power-of-two-sized.
 */
struct padded_float {
  float value_;
  int padding1;
  int padding2;

  padded_float() = default;
  constexpr padded_float(const float& x) : value_(x), padding1(0), padding2(0) {}
  constexpr padded_float(const padded_float&)            = default;
  constexpr padded_float& operator=(const padded_float&) = default;
  constexpr float abs() const { return std::abs(value_); }
};

constexpr padded_float operator+(const padded_float& x, const padded_float& y)
{
  return padded_float(x.value_ + y.value_);
}

constexpr padded_float operator-(const padded_float& x, const padded_float& y)
{
  return padded_float(x.value_ - y.value_);
}
constexpr padded_float operator*(const padded_float& x, const padded_float& y)
{
  return padded_float(x.value_ * y.value_);
}
constexpr padded_float operator*(const padded_float& x, const int& scalar)
{
  return padded_float(scalar * x.value_);
}
constexpr bool operator==(const padded_float& x, const padded_float& y)
{
  return x.value_ == y.value_;
}

constexpr bool operator<(const padded_float& x, const padded_float& y)
{
  return x.value_ < y.value_;
}
constexpr bool operator>(const padded_float& x, const padded_float& y)
{
  return x.value_ > y.value_;
}
inline auto operator<<(std::ostream& os, const padded_float& x) -> std::ostream&
{
  os << x.value_;
  return os;
}

template <typename InType, typename IdxType, typename OutType>
void mapLaunch(OutType* out,
               const InType* in1,
               const InType* in2,
               const InType* in3,
               InType scalar,
               IdxType len,
               cudaStream_t stream)
{
  raft::resources handle;
  resource::set_cuda_stream(handle, stream);
  auto out_view = raft::make_device_vector_view(out, len);
  auto in1_view = raft::make_device_vector_view(in1, len);
  auto in2_view = raft::make_device_vector_view(in2, len);
  auto in3_view = raft::make_device_vector_view(in3, len);
  map(
    handle,
    out_view,
    [=] __device__(InType a, InType b, InType c) { return a + b + c + scalar; },
    in1_view,
    in2_view,
    in3_view);
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct MapInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
  InType scalar;
};

template <typename InType, typename IdxType, typename OutType = InType>
void create_ref(OutType* out_ref,
                const InType* in1,
                const InType* in2,
                const InType* in3,
                InType scalar,
                IdxType len,
                cudaStream_t stream)
{
  rmm::device_uvector<InType> tmp(len, stream);
  eltwiseAdd(tmp.data(), in1, in2, len, stream);
  eltwiseAdd(out_ref, tmp.data(), in3, len, stream);
  scalarAdd(out_ref, out_ref, (OutType)scalar, len, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename InType, typename IdxType, typename OutType = InType>
class MapTest : public ::testing::TestWithParam<MapInputs<InType, IdxType, OutType>> {
 public:
  MapTest()
    : params(::testing::TestWithParam<MapInputs<InType, IdxType, OutType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in1(params.len, stream),
      in2(params.len, stream),
      in3(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);

    IdxType len = params.len;
    if constexpr (std::is_floating_point<InType>::value) {
      uniform(handle, r, in1.data(), len, InType(-1.0), InType(1.0));
      uniform(handle, r, in2.data(), len, InType(-1.0), InType(1.0));
      uniform(handle, r, in3.data(), len, InType(-1.0), InType(1.0));
    } else {
      // First create random float arrays
      rmm::device_uvector<float> fin1(params.len, stream);
      rmm::device_uvector<float> fin2(params.len, stream);
      rmm::device_uvector<float> fin3(params.len, stream);
      uniform(handle, r, fin1.data(), len, float(-1.0), float(1.0));
      uniform(handle, r, fin2.data(), len, float(-1.0), float(1.0));
      uniform(handle, r, fin3.data(), len, float(-1.0), float(1.0));

      // Then pad them
      raft::device_resources handle{stream};
      auto fin1_view = raft::make_device_vector_view(fin1.data(), fin1.size());
      auto fin2_view = raft::make_device_vector_view(fin2.data(), fin2.size());
      auto fin3_view = raft::make_device_vector_view(fin3.data(), fin3.size());
      auto in1_view  = raft::make_device_vector_view(in1.data(), in1.size());
      auto in2_view  = raft::make_device_vector_view(in2.data(), in2.size());
      auto in3_view  = raft::make_device_vector_view(in3.data(), in3.size());

      auto add_padding = [] __device__(float a) { return padded_float(a); };
      raft::linalg::map(handle, in1_view, add_padding, raft::make_const_mdspan(fin1_view));
      raft::linalg::map(handle, in2_view, add_padding, raft::make_const_mdspan(fin2_view));
      raft::linalg::map(handle, in3_view, add_padding, raft::make_const_mdspan(fin3_view));
    }

    create_ref(out_ref.data(), in1.data(), in2.data(), in3.data(), params.scalar, len, stream);
    mapLaunch(out.data(), in1.data(), in2.data(), in3.data(), params.scalar, len, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MapInputs<InType, IdxType, OutType> params;
  rmm::device_uvector<InType> in1, in2, in3;
  rmm::device_uvector<OutType> out_ref, out;
};

template <typename OutType, typename IdxType>
class MapOffsetTest : public ::testing::TestWithParam<MapInputs<OutType, IdxType, OutType>> {
 public:
  MapOffsetTest()
    : params(::testing::TestWithParam<MapInputs<OutType, IdxType, OutType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    IdxType len    = params.len;
    OutType scalar = params.scalar;
    naiveScale(out_ref.data(), (OutType*)nullptr, scalar, len, stream);

    auto out_view = raft::make_device_vector_view(out.data(), len);
    map_offset(handle,
               out_view,
               raft::compose_op(raft::cast_op<OutType>(), raft::mul_const_op<OutType>(scalar)));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MapInputs<OutType, IdxType, OutType> params;
  rmm::device_uvector<OutType> out_ref, out;
};

#define MAP_TEST(test_type, test_name, inputs)                       \
  typedef RAFT_DEPAREN(test_type) test_name;                         \
  TEST_P(test_name, Result)                                          \
  {                                                                  \
    ASSERT_TRUE(devArrMatch(this->out_ref.data(),                    \
                            this->out.data(),                        \
                            this->params.len,                        \
                            CompareApprox(this->params.tolerance))); \
  }                                                                  \
  INSTANTIATE_TEST_SUITE_P(MapTests, test_name, ::testing::ValuesIn(inputs))

const std::vector<MapInputs<float, int>> inputsf_i32 = {{0.000001f, 1024 * 1024, 1234ULL, 3.2}};
MAP_TEST((MapTest<float, int>), MapTestF_i32, inputsf_i32);
MAP_TEST((MapOffsetTest<float, int>), MapOffsetTestF_i32, inputsf_i32);

const std::vector<MapInputs<float, size_t>> inputsf_i64 = {{0.000001f, 1024 * 1024, 1234ULL, 9.4}};
MAP_TEST((MapTest<float, size_t>), MapTestF_i64, inputsf_i64);
MAP_TEST((MapOffsetTest<float, size_t>), MapOffsetTestF_i64, inputsf_i64);

const std::vector<MapInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 1234ULL, 5.9}};
MAP_TEST((MapTest<float, int, double>), MapTestF_i32_D, inputsf_i32_d);

const std::vector<MapInputs<double, int>> inputsd_i32 = {{0.00000001, 1024 * 1024, 1234ULL, 7.5}};
MAP_TEST((MapTest<double, int>), MapTestD_i32, inputsd_i32);
MAP_TEST((MapOffsetTest<double, int>), MapOffsetTestD_i32, inputsd_i32);

const std::vector<MapInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 1234ULL, 5.2}};
MAP_TEST((MapTest<double, size_t>), MapTestD_i64, inputsd_i64);
MAP_TEST((MapOffsetTest<double, size_t>), MapOffsetTestD_i64, inputsd_i64);

// This comparison structure is necessary, because it is not straight-forward to
// add an overload of std::abs for padded_float.
struct ComparePadded {
  float eps;
  ComparePadded(float eps_) : eps(eps_) {}
  ComparePadded(padded_float eps_) : eps(eps_.value_) {}
  ComparePadded(double eps_) : eps(eps_) {}
  bool operator()(const padded_float& a, const padded_float& b) const
  {
    float diff  = (a - b).abs();
    float m     = std::max(a.abs(), b.abs());
    float ratio = diff > eps ? diff / m : diff;
    return (ratio <= eps);
  }
};

// Use PaddedComparison
#define MAP_TEST_PADDED(test_type, test_name, inputs)                \
  typedef RAFT_DEPAREN(test_type) test_name;                         \
  TEST_P(test_name, Result)                                          \
  {                                                                  \
    ASSERT_TRUE(devArrMatch(this->out_ref.data(),                    \
                            this->out.data(),                        \
                            this->params.len,                        \
                            ComparePadded(this->params.tolerance))); \
  }                                                                  \
  INSTANTIATE_TEST_SUITE_P(MapTests, test_name, ::testing::ValuesIn(inputs))

const std::vector<MapInputs<padded_float, size_t>> inputsd_padded_float = {
  {0.00000001, 1024 * 1024, 1234ULL, 5.2}};
MAP_TEST_PADDED((MapTest<padded_float, size_t>), MapTestD_padded_float, inputsd_padded_float);
MAP_TEST_PADDED((MapOffsetTest<padded_float, size_t>),
                MapOffsetTestD_padded_float,
                inputsd_padded_float);

}  // namespace linalg
}  // namespace raft
