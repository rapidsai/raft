/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#pragma once

#include "processing.hpp"

#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>

#include <rmm/device_uvector.hpp>

namespace raft {
namespace spatial {
namespace knn {

template <typename math_t>
class CosineMetricProcessor : public MetricProcessor<math_t> {
 protected:
  int k_;
  bool row_major_;
  size_t n_rows_;
  size_t n_cols_;
  cudaStream_t stream_;
  rmm::device_uvector<math_t> colsums_;

 public:
  CosineMetricProcessor(size_t n_rows, size_t n_cols, int k, bool row_major, cudaStream_t stream)
    : stream_(stream),
      colsums_(n_rows, stream),
      n_cols_(n_cols),
      n_rows_(n_rows),
      row_major_(row_major),
      k_(k)
  {
  }

  void preprocess(math_t* data)
  {
    raft::linalg::rowNorm(colsums_.data(),
                          data,
                          n_cols_,
                          n_rows_,
                          raft::linalg::NormType::L2Norm,
                          row_major_,
                          stream_,
                          raft::sqrt_op{});

    raft::linalg::matrixVectorOp(
      data, data, colsums_.data(), n_cols_, n_rows_, row_major_, false, raft::div_op{}, stream_);
  }

  void revert(math_t* data)
  {
    raft::linalg::matrixVectorOp(
      data, data, colsums_.data(), n_cols_, n_rows_, row_major_, false, raft::mul_op{}, stream_);
  }

  void postprocess(math_t* data)
  {
    raft::linalg::unaryOp(
      data, data, k_ * n_rows_, [] __device__(math_t in) { return 1 - in; }, stream_);
  }

  void set_num_queries(int k) override { k_ = k; }

  ~CosineMetricProcessor() = default;
};

template <typename math_t>
class CorrelationMetricProcessor : public CosineMetricProcessor<math_t> {
  using cosine = CosineMetricProcessor<math_t>;

 public:
  CorrelationMetricProcessor(
    size_t n_rows, size_t n_cols, int k, bool row_major, cudaStream_t stream)
    : CosineMetricProcessor<math_t>(n_rows, n_cols, k, row_major, stream), means_(n_rows, stream)
  {
  }

  void preprocess(math_t* data)
  {
    math_t normalizer_const = 1.0 / (math_t)cosine::n_cols_;

    raft::linalg::reduce(means_.data(),
                         data,
                         cosine::n_cols_,
                         cosine::n_rows_,
                         (math_t)0.0,
                         cosine::row_major_,
                         true,
                         cosine::stream_);

    raft::linalg::unaryOp(means_.data(),
                          means_.data(),
                          cosine::n_rows_,
                          raft::mul_const_op<math_t>(normalizer_const),
                          cosine::stream_);

    raft::stats::meanCenter(data,
                            data,
                            means_.data(),
                            cosine::n_cols_,
                            cosine::n_rows_,
                            cosine::row_major_,
                            false,
                            cosine::stream_);

    CosineMetricProcessor<math_t>::preprocess(data);
  }

  void revert(math_t* data)
  {
    CosineMetricProcessor<math_t>::revert(data);

    raft::stats::meanAdd(data,
                         data,
                         means_.data(),
                         cosine::n_cols_,
                         cosine::n_rows_,
                         cosine::row_major_,
                         false,
                         cosine::stream_);
  }

  void postprocess(math_t* data) { CosineMetricProcessor<math_t>::postprocess(data); }

  ~CorrelationMetricProcessor() = default;

  rmm::device_uvector<math_t> means_;
};

template <typename math_t>
class DefaultMetricProcessor : public MetricProcessor<math_t> {
 public:
  void preprocess(math_t* data) {}

  void revert(math_t* data) {}

  void postprocess(math_t* data) {}

  ~DefaultMetricProcessor() = default;
};

template <typename math_t>
inline std::unique_ptr<MetricProcessor<math_t>> create_processor(
  distance::DistanceType metric, int n, int D, int k, bool rowMajorQuery, cudaStream_t userStream)
{
  MetricProcessor<math_t>* mp = nullptr;

  switch (metric) {
    case distance::DistanceType::CosineExpanded:
      mp = new CosineMetricProcessor<math_t>(n, D, k, rowMajorQuery, userStream);
      break;

    case distance::DistanceType::CorrelationExpanded:
      mp = new CorrelationMetricProcessor<math_t>(n, D, k, rowMajorQuery, userStream);
      break;
    default: mp = new DefaultMetricProcessor<math_t>();
  }

  return std::unique_ptr<MetricProcessor<math_t>>(mp);
}

// Currently only being used by floats
template class MetricProcessor<float>;
template class CosineMetricProcessor<float>;
template class CorrelationMetricProcessor<float>;
template class DefaultMetricProcessor<float>;

}  // namespace knn
}  // namespace spatial
}  // namespace raft
