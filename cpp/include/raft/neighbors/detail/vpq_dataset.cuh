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

#include <raft/cluster/kmeans_balanced.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/neighbors/detail/ivf_pq_build.cuh>  // pq_bits-bitfield
#include <raft/spatial/knn/detail/ann_utils.cuh>   // utils::mapping etc
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

// A temporary stub till https://github.com/rapidsai/raft/pull/2077 is re-merged
namespace raft::util {

/**
 * Subsample the dataset to create a training set.
 *
 * @tparam DatasetT a row-major mdspan or mdarray (device or host)
 *
 * @param res raft handle
 * @param dataset input row-major mdspan or mdarray (device or host)
 * @param n_samples the size of the output mdarray
 *
 * @return a newly allocated subset of the dataset.
 */
template <typename DatasetT>
auto subsample(raft::resources const& res,
               const DatasetT& dataset,
               typename DatasetT::index_type n_samples)
  -> raft::device_matrix<typename DatasetT::value_type, typename DatasetT::index_type>
{
  using value_type = typename DatasetT::value_type;
  using index_type = typename DatasetT::index_type;
  static_assert(std::is_same_v<typename DatasetT::layout_type, row_major>,
                "Only row-major layout is supported at the moment");
  RAFT_EXPECTS(n_samples <= dataset.extent(0),
               "The number of samples must be smaller than the number of input rows in the current "
               "implementation.");
  size_t dim            = dataset.extent(1);
  size_t trainset_ratio = dataset.extent(0) / n_samples;
  auto result = raft::make_device_matrix<value_type, index_type>(res, n_samples, dataset.extent(1));

  RAFT_CUDA_TRY(cudaMemcpy2DAsync(result.data_handle(),
                                  sizeof(value_type) * dim,
                                  dataset.data_handle(),
                                  sizeof(value_type) * dim * trainset_ratio,
                                  sizeof(value_type) * dim,
                                  n_samples,
                                  cudaMemcpyDefault,
                                  raft::resource::get_cuda_stream(res)));
  return result;
}

}  // namespace raft::util

namespace raft::neighbors::detail {

template <typename DatasetT>
auto fill_missing_params_heuristics(const vpq_params& params, const DatasetT& dataset) -> vpq_params
{
  vpq_params r  = params;
  double n_rows = dataset.extent(0);
  size_t dim    = dataset.extent(1);
  if (r.pq_dim == 0) { r.pq_dim = raft::div_rounding_up_safe(dim, size_t{4}); }
  if (r.pq_bits == 0) { r.pq_bits = 8; }
  if (r.vq_n_centers == 0) { r.vq_n_centers = raft::round_up_safe<uint32_t>(std::sqrt(n_rows), 8); }
  if (r.vq_kmeans_trainset_fraction == 0) {
    double vq_trainset_size       = 100.0 * r.vq_n_centers;
    r.vq_kmeans_trainset_fraction = std::min(1.0, vq_trainset_size / n_rows);
  }
  if (r.pq_kmeans_trainset_fraction == 0) {
    // NB: we'll have actually `pq_dim` times more samples than this
    //     (because the dataset is reinterpreted as `[n_rows * pq_dim, pq_len]`)
    double pq_trainset_size       = 1000.0 * (1u << r.pq_bits);
    r.pq_kmeans_trainset_fraction = std::min(1.0, pq_trainset_size / n_rows);
  }
  return r;
}

template <typename T, typename DatasetT>
auto transform_data(const raft::resources& res, DatasetT dataset)
  -> device_mdarray<T, typename DatasetT::extents_type, typename DatasetT::layout_type>
{
  using index_type       = typename DatasetT::index_type;
  using extents_type     = typename DatasetT::extents_type;
  using layout_type      = typename DatasetT::layout_type;
  using out_mdarray_type = device_mdarray<T, extents_type, layout_type>;
  if constexpr (std::is_same_v<out_mdarray_type, std::decay<DatasetT>>) { return dataset; }

  auto result = raft::make_device_mdarray<T, index_type, layout_type>(res, dataset.extents());

  linalg::map(res,
              result.view(),
              spatial::knn::detail::utils::mapping<T>{},
              raft::make_const_mdspan(dataset.view()));

  return result;
}

/** Fix the internal indexing type to avoid integer underflows/overflows */
using ix_t = int64_t;

template <typename MathT, typename DatasetT>
auto train_vq(const raft::resources& res, const vpq_params& params, const DatasetT& dataset)
  -> device_matrix<MathT, uint32_t, row_major>
{
  const ix_t n_rows       = dataset.extent(0);
  const ix_t vq_n_centers = params.vq_n_centers;
  const ix_t dim          = dataset.extent(1);
  const ix_t n_rows_train = n_rows * params.vq_kmeans_trainset_fraction;

  // Subsample the dataset and transform into the required type if necessary
  auto vq_trainset = raft::util::subsample(res, dataset, n_rows_train);
  auto vq_centers  = raft::make_device_matrix<MathT, uint32_t, row_major>(res, vq_n_centers, dim);

  using kmeans_in_type = typename DatasetT::value_type;
  raft::cluster::kmeans_balanced_params kmeans_params;
  kmeans_params.n_iters = params.kmeans_n_iters;
  kmeans_params.metric  = raft::distance::DistanceType::L2Expanded;
  auto vq_centers_view =
    raft::make_device_matrix_view<MathT, ix_t>(vq_centers.data_handle(), vq_n_centers, dim);
  auto vq_trainset_view = raft::make_device_matrix_view<const kmeans_in_type, ix_t>(
    vq_trainset.data_handle(), n_rows_train, dim);
  raft::cluster::kmeans_balanced::fit<kmeans_in_type, MathT, ix_t>(
    res,
    kmeans_params,
    vq_trainset_view,
    vq_centers_view,
    spatial::knn::detail::utils::mapping<MathT>{});

  return vq_centers;
}

template <typename LabelT, typename DatasetT, typename VqCentersT>
auto predict_vq(const raft::resources& res, const DatasetT& dataset, const VqCentersT& vq_centers)
  -> device_vector<LabelT, typename DatasetT::index_type>
{
  using kmeans_data_type = typename DatasetT::value_type;
  using kmeans_math_type = typename VqCentersT::value_type;
  using index_type       = typename DatasetT::index_type;
  using label_type       = LabelT;

  auto vq_labels = raft::make_device_vector<label_type, index_type>(res, dataset.extent(0));

  raft::cluster::kmeans_balanced_params kmeans_params;
  kmeans_params.metric = raft::distance::DistanceType::L2Expanded;

  auto vq_centers_view = raft::make_device_matrix_view<const kmeans_math_type, index_type>(
    vq_centers.data_handle(), vq_centers.extent(0), vq_centers.extent(1));

  auto vq_dataset_view = raft::make_device_matrix_view<const kmeans_data_type, index_type>(
    dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  raft::cluster::kmeans_balanced::
    predict<kmeans_data_type, kmeans_math_type, index_type, label_type>(
      res,
      kmeans_params,
      vq_dataset_view,
      vq_centers_view,
      vq_labels.view(),
      spatial::knn::detail::utils::mapping<kmeans_math_type>{});

  return vq_labels;
}

template <typename MathT, typename DatasetT>
auto train_pq(const raft::resources& res,
              const vpq_params& params,
              const DatasetT& dataset,
              const device_matrix_view<const MathT, uint32_t, row_major>& vq_centers)
  -> device_matrix<MathT, uint32_t, row_major>
{
  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  const ix_t n_rows_train = n_rows * params.pq_kmeans_trainset_fraction;

  // Subsample the dataset and transform into the required type if necessary
  auto pq_trainset = transform_data<MathT>(res, raft::util::subsample(res, dataset, n_rows_train));

  // Subtract VQ centers
  {
    auto vq_labels   = predict_vq<uint32_t>(res, pq_trainset, vq_centers);
    using index_type = typename DatasetT::index_type;
    linalg::map_offset(
      res,
      pq_trainset.view(),
      [labels = vq_labels.view(), centers = vq_centers, dim] __device__(index_type off, MathT x) {
        index_type i = off / dim;
        index_type j = off % dim;
        return x - centers(labels(i), j);
      },
      raft::make_const_mdspan(pq_trainset.view()));
  }

  auto pq_centers = raft::make_device_matrix<MathT, uint32_t, row_major>(res, pq_n_centers, pq_len);

  // Train PQ centers
  {
    raft::cluster::kmeans_balanced_params kmeans_params;
    kmeans_params.n_iters = params.kmeans_n_iters;
    kmeans_params.metric  = raft::distance::DistanceType::L2Expanded;

    auto pq_centers_view =
      raft::make_device_matrix_view<MathT, ix_t>(pq_centers.data_handle(), pq_n_centers, pq_len);

    auto pq_trainset_view = raft::make_device_matrix_view<const MathT, ix_t>(
      pq_trainset.data_handle(), n_rows_train * pq_dim, pq_len);

    raft::cluster::kmeans_balanced::fit<MathT, MathT, ix_t>(
      res, kmeans_params, pq_trainset_view, pq_centers_view);
  }

  return pq_centers;
}

template <uint32_t SubWarpSize, typename DataT, typename MathT, typename IdxT, typename LabelT>
__device__ auto compute_code(device_matrix_view<const DataT, IdxT, row_major> dataset,
                             device_matrix_view<const MathT, uint32_t, row_major> vq_centers,
                             device_matrix_view<const MathT, uint32_t, row_major> pq_centers,
                             IdxT i,
                             uint32_t j,
                             LabelT vq_label) -> uint8_t
{
  auto data_mapping = spatial::knn::detail::utils::mapping<MathT>{};
  uint32_t lane_id  = Pow2<SubWarpSize>::mod(laneId());

  const uint32_t pq_book_size = pq_centers.extent(0);
  const uint32_t pq_len       = pq_centers.extent(1);
  float min_dist              = std::numeric_limits<float>::infinity();
  uint8_t code                = 0;
  // calculate the distance for each PQ cluster, find the minimum for each thread
  for (uint32_t l = lane_id; l < pq_book_size; l += SubWarpSize) {
    // NB: the L2 quantifiers on residuals are always trained on L2 metric.
    float d = 0.0f;
    for (uint32_t k = 0; k < pq_len; k++) {
      auto jk = j * pq_len + k;
      auto x  = data_mapping(dataset(i, jk)) - vq_centers(vq_label, jk);
      auto t  = x - pq_centers(l, k);
      d += t * t;
    }
    if (d < min_dist) {
      min_dist = d;
      code     = uint8_t(l);
    }
  }
  // reduce among threads
#pragma unroll
  for (uint32_t stride = SubWarpSize >> 1; stride > 0; stride >>= 1) {
    const auto other_dist = shfl_xor(min_dist, stride, SubWarpSize);
    const auto other_code = shfl_xor(code, stride, SubWarpSize);
    if (other_dist < min_dist) {
      min_dist = other_dist;
      code     = other_code;
    }
  }
  return code;
}

template <uint32_t BlockSize,
          uint32_t PqBits,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL
  process_and_fill_codes_kernel(device_matrix_view<uint8_t, IdxT, row_major> out_codes,
                                device_matrix_view<const DataT, IdxT, row_major> dataset,
                                device_matrix_view<const MathT, uint32_t, row_major> vq_centers,
                                device_vector_view<const LabelT, IdxT, row_major> vq_labels,
                                device_matrix_view<const MathT, uint32_t, row_major> pq_centers)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(WarpSize, 1u << PqBits);
  using subwarp_align             = Pow2<kSubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t pq_dim = raft::div_rounding_up_unsafe(vq_centers.extent(1), pq_centers.extent(1));

  const uint32_t lane_id = Pow2<kSubWarpSize>::mod(threadIdx.x);
  const LabelT vq_label  = vq_labels(row_ix);

  // write label
  auto* out_label_ptr = reinterpret_cast<LabelT*>(&out_codes(row_ix, 0));
  if (lane_id == 0) { *out_label_ptr = vq_label; }

  auto* out_codes_ptr = reinterpret_cast<uint8_t*>(out_label_ptr + 1);
  ivf_pq::detail::bitfield_view_t<PqBits> code_view{out_codes_ptr};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    uint8_t code = compute_code<kSubWarpSize>(dataset, vq_centers, pq_centers, row_ix, j, vq_label);
    // TODO: this writes in global memory one byte per warp, which is very slow.
    //  It's better to keep the codes in the shared memory or registers and dump them at once.
    if (lane_id == 0) { code_view[j] = code; }
  }
}

template <typename MathT, typename IdxT, typename DatasetT>
auto process_and_fill_codes(const raft::resources& res,
                            const vpq_params& params,
                            const DatasetT& dataset,
                            device_matrix_view<const MathT, uint32_t, row_major> vq_centers,
                            device_matrix_view<const MathT, uint32_t, row_major> pq_centers)
  -> device_matrix<uint8_t, IdxT, row_major>
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  // NB: codes must be aligned at least to sizeof(label_t) to be able to read labels.
  const ix_t codes_rowlen =
    sizeof(label_t) * (1 + raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8 * sizeof(label_t)));

  auto codes = raft::make_device_matrix<uint8_t, IdxT, row_major>(res, n_rows, codes_rowlen);

  auto stream = raft::resource::get_cuda_stream(res);

  // TODO: with scaling workspace we could choose the batch size dynamically
  constexpr ix_t kReasonableMaxBatchSize = 65536;
  constexpr ix_t kBlockSize              = 256;
  const ix_t threads_per_vec             = std::min<ix_t>(WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);
  ix_t max_batch_size = std::min<ix_t>(n_rows, kReasonableMaxBatchSize);
  auto kernel         = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return process_and_fill_codes_kernel<kBlockSize, 4, data_t, MathT, IdxT, label_t>;
      case 5: return process_and_fill_codes_kernel<kBlockSize, 5, data_t, MathT, IdxT, label_t>;
      case 6: return process_and_fill_codes_kernel<kBlockSize, 6, data_t, MathT, IdxT, label_t>;
      case 7: return process_and_fill_codes_kernel<kBlockSize, 7, data_t, MathT, IdxT, label_t>;
      case 8: return process_and_fill_codes_kernel<kBlockSize, 8, data_t, MathT, IdxT, label_t>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(pq_bits);
  for (const auto& batch :
       spatial::knn::detail::utils::batch_load_iterator(dataset.data_handle(),
                                                        n_rows,
                                                        dim,
                                                        max_batch_size,
                                                        stream,
                                                        rmm::mr::get_current_device_resource())) {
    auto batch_view = raft::make_device_matrix_view(batch.data(), ix_t(batch.size()), dim);
    auto labels     = predict_vq<label_t>(res, batch_view, vq_centers);
    dim3 blocks(div_rounding_up_safe<ix_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
    kernel<<<blocks, threads, 0, stream>>>(
      make_device_matrix_view<uint8_t, IdxT>(
        codes.data_handle() + batch.offset() * codes_rowlen, batch.size(), codes_rowlen),
      batch_view,
      vq_centers,
      make_const_mdspan(labels.view()),
      pq_centers);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  return codes;
}

template <typename NewMathT, typename OldMathT, typename IdxT>
auto vpq_convert_math_type(const raft::resources& res, vpq_dataset<OldMathT, IdxT>&& src)
  -> vpq_dataset<NewMathT, IdxT>
{
  auto vq_code_book = make_device_mdarray<NewMathT>(res, src.vq_code_book.extents());
  auto pq_code_book = make_device_mdarray<NewMathT>(res, src.pq_code_book.extents());

  linalg::map(res,
              vq_code_book.view(),
              spatial::knn::detail::utils::mapping<NewMathT>{},
              raft::make_const_mdspan(src.vq_code_book.view()));
  linalg::map(res,
              pq_code_book.view(),
              spatial::knn::detail::utils::mapping<NewMathT>{},
              raft::make_const_mdspan(src.pq_code_book.view()));
  return vpq_dataset<NewMathT, IdxT>{
    std::move(vq_code_book), std::move(pq_code_book), std::move(src.data)};
}

template <typename DatasetT, typename MathT, typename IdxT>
auto vpq_build(const raft::resources& res, const vpq_params& params, const DatasetT& dataset)
  -> vpq_dataset<MathT, IdxT>
{
  // Use a heuristic to impute missing parameters.
  auto ps = fill_missing_params_heuristics(params, dataset);

  // Train codes
  auto vq_code_book = train_vq<MathT>(res, ps, dataset);
  auto pq_code_book =
    train_pq<MathT>(res, ps, dataset, raft::make_const_mdspan(vq_code_book.view()));

  // Encode dataset
  auto codes = process_and_fill_codes<MathT, IdxT>(res,
                                                   ps,
                                                   dataset,
                                                   raft::make_const_mdspan(vq_code_book.view()),
                                                   raft::make_const_mdspan(pq_code_book.view()));

  return vpq_dataset<MathT, IdxT>{
    std::move(vq_code_book), std::move(pq_code_book), std::move(codes)};
}

}  // namespace raft::neighbors::detail
