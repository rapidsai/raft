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
#pragma once

#include "../dataset.hpp"

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>

#include <cuda_fp16.h>

#include <fstream>
#include <memory>

namespace raft::neighbors::detail {

using dataset_instance_tag                              = uint32_t;
constexpr dataset_instance_tag kSerializeEmptyDataset   = 1;
constexpr dataset_instance_tag kSerializeStridedDataset = 2;
constexpr dataset_instance_tag kSerializeVPQDataset     = 3;

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const empty_dataset<IdxT>& dataset)
{
  serialize_scalar(res, os, dataset.suggested_dim);
}

template <typename DataT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const strided_dataset<DataT, IdxT>& dataset)
{
  auto n_rows = dataset.n_rows();
  auto dim    = dataset.dim();
  auto stride = dataset.stride();
  serialize_scalar(res, os, n_rows);
  serialize_scalar(res, os, dim);
  serialize_scalar(res, os, stride);
  // Remove padding before saving the dataset
  auto src = dataset.view();
  auto dst = make_host_matrix<DataT, IdxT>(n_rows, dim);
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(dst.data_handle(),
                                  sizeof(DataT) * dim,
                                  src.data_handle(),
                                  sizeof(DataT) * stride,
                                  sizeof(DataT) * dim,
                                  n_rows,
                                  cudaMemcpyDefault,
                                  resource::get_cuda_stream(res)));
  resource::sync_stream(res);
  serialize_mdspan(res, os, dst.view());
}

template <typename MathT, typename IdxT>
void serialize(const raft::resources& res,
               std::ostream& os,
               const vpq_dataset<MathT, IdxT>& dataset)
{
  serialize_scalar(res, os, dataset.n_rows());
  serialize_scalar(res, os, dataset.dim());
  serialize_scalar(res, os, dataset.vq_n_centers());
  serialize_scalar(res, os, dataset.pq_n_centers());
  serialize_scalar(res, os, dataset.pq_len());
  serialize_scalar(res, os, dataset.encoded_row_length());
  serialize_mdspan(res, os, make_const_mdspan(dataset.vq_code_book.view()));
  serialize_mdspan(res, os, make_const_mdspan(dataset.pq_code_book.view()));
  serialize_mdspan(res, os, make_const_mdspan(dataset.data.view()));
}

template <typename IdxT>
void serialize(const raft::resources& res, std::ostream& os, const dataset<IdxT>& dataset)
{
  if (auto x = dynamic_cast<const empty_dataset<IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeEmptyDataset);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<float, IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeStridedDataset);
    serialize_scalar(res, os, CUDA_R_32F);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<half, IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeStridedDataset);
    serialize_scalar(res, os, CUDA_R_16F);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<int8_t, IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeStridedDataset);
    serialize_scalar(res, os, CUDA_R_8I);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const strided_dataset<uint8_t, IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeStridedDataset);
    serialize_scalar(res, os, CUDA_R_8U);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const vpq_dataset<float, IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeVPQDataset);
    serialize_scalar(res, os, CUDA_R_32F);
    return serialize(res, os, *x);
  }
  if (auto x = dynamic_cast<const vpq_dataset<half, IdxT>*>(&dataset); x != nullptr) {
    serialize_scalar(res, os, kSerializeVPQDataset);
    serialize_scalar(res, os, CUDA_R_16F);
    return serialize(res, os, *x);
  }
  RAFT_FAIL("unsupported dataset type.");
}

template <typename IdxT>
auto deserialize_empty(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<empty_dataset<IdxT>>
{
  auto suggested_dim = deserialize_scalar<uint32_t>(res, is);
  return std::make_unique<empty_dataset<IdxT>>(suggested_dim);
}

template <typename DataT, typename IdxT>
auto deserialize_strided(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<strided_dataset<DataT, IdxT>>
{
  auto n_rows     = deserialize_scalar<IdxT>(res, is);
  auto dim        = deserialize_scalar<uint32_t>(res, is);
  auto stride     = deserialize_scalar<uint32_t>(res, is);
  auto host_array = make_host_matrix<DataT, IdxT>(n_rows, dim);
  deserialize_mdspan(res, is, host_array.view());
  return make_strided_dataset(res, host_array, stride);
}

template <typename MathT, typename IdxT>
auto deserialize_vpq(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<vpq_dataset<MathT, IdxT>>
{
  auto n_rows             = deserialize_scalar<IdxT>(res, is);
  auto dim                = deserialize_scalar<uint32_t>(res, is);
  auto vq_n_centers       = deserialize_scalar<uint32_t>(res, is);
  auto pq_n_centers       = deserialize_scalar<uint32_t>(res, is);
  auto pq_len             = deserialize_scalar<uint32_t>(res, is);
  auto encoded_row_length = deserialize_scalar<uint32_t>(res, is);

  auto vq_code_book = make_device_matrix<MathT, uint32_t, row_major>(res, vq_n_centers, dim);
  auto pq_code_book = make_device_matrix<MathT, uint32_t, row_major>(res, pq_n_centers, pq_len);
  auto data         = make_device_matrix<uint8_t, IdxT, row_major>(res, n_rows, encoded_row_length);

  deserialize_mdspan(res, is, vq_code_book.view());
  deserialize_mdspan(res, is, pq_code_book.view());
  deserialize_mdspan(res, is, data.view());

  return std::make_unique<vpq_dataset<MathT, IdxT>>(
    std::move(vq_code_book), std::move(pq_code_book), std::move(data));
}

template <typename IdxT>
auto deserialize_dataset(raft::resources const& res, std::istream& is)
  -> std::unique_ptr<dataset<IdxT>>
{
  switch (deserialize_scalar<dataset_instance_tag>(res, is)) {
    case kSerializeEmptyDataset: return deserialize_empty<IdxT>(res, is);
    case kSerializeStridedDataset:
      switch (deserialize_scalar<cudaDataType_t>(res, is)) {
        case CUDA_R_32F: return deserialize_strided<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_strided<half, IdxT>(res, is);
        case CUDA_R_8I: return deserialize_strided<int8_t, IdxT>(res, is);
        case CUDA_R_8U: return deserialize_strided<uint8_t, IdxT>(res, is);
        default: break;
      }
    case kSerializeVPQDataset:
      switch (deserialize_scalar<cudaDataType_t>(res, is)) {
        case CUDA_R_32F: return deserialize_vpq<float, IdxT>(res, is);
        case CUDA_R_16F: return deserialize_vpq<half, IdxT>(res, is);
        default: break;
      }
    default: break;
  }
  RAFT_FAIL("Failed to deserialize dataset: unsupported combination of instance tags.");
}

}  // namespace raft::neighbors::detail
