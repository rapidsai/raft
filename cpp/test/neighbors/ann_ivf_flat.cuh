/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "ann_utils.cuh"
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>

#include <raft_internal/neighbors/naive_knn.cuh>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/ann.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/stats/mean.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <thrust/sequence.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft::neighbors::ivf_flat {

template <typename IdxT>
struct AnnIvfFlatInputs {
  IdxT num_queries;
  IdxT num_db_vecs;
  IdxT dim;
  IdxT k;
  IdxT nprobe;
  IdxT nlist;
  raft::distance::DistanceType metric;
  bool adaptive_centers;
};

template <typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const AnnIvfFlatInputs<IdxT>& p)
{
  os << "{ " << p.num_queries << ", " << p.num_db_vecs << ", " << p.dim << ", " << p.k << ", "
     << p.nprobe << ", " << p.nlist << ", " << static_cast<int>(p.metric) << ", "
     << p.adaptive_centers << '}' << std::endl;
  return os;
}

template <typename T, typename DataT, typename IdxT>
class AnnIVFFlatTest : public ::testing::TestWithParam<AnnIvfFlatInputs<IdxT>> {
 public:
  AnnIVFFlatTest()
    : stream_(resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnIvfFlatInputs<IdxT>>::GetParam()),
      database(0, stream_),
      search_queries(0, stream_)
  {
  }

 protected:
  void testIVFFlat()
  {
    size_t queries_size = ps.num_queries * ps.k;
    std::vector<IdxT> indices_ivfflat(queries_size);
    std::vector<IdxT> indices_naive(queries_size);
    std::vector<T> distances_ivfflat(queries_size);
    std::vector<T> distances_naive(queries_size);

    {
      rmm::device_uvector<T> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_naive_dev(queries_size, stream_);
      naive_knn<T, DataT, IdxT>(handle_,
                                distances_naive_dev.data(),
                                indices_naive_dev.data(),
                                search_queries.data(),
                                database.data(),
                                ps.num_queries,
                                ps.num_db_vecs,
                                ps.dim,
                                ps.k,
                                ps.metric);
      update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      resource::sync_stream(handle_);
    }

    {
      // unless something is really wrong with clustering, this could serve as a lower bound on
      // recall
      double min_recall = static_cast<double>(ps.nprobe) / static_cast<double>(ps.nlist);

      rmm::device_uvector<T> distances_ivfflat_dev(queries_size, stream_);
      rmm::device_uvector<IdxT> indices_ivfflat_dev(queries_size, stream_);

      {
        // legacy interface
        raft::spatial::knn::IVFFlatParam ivfParams;
        ivfParams.nprobe = ps.nprobe;
        ivfParams.nlist  = ps.nlist;
        raft::spatial::knn::knnIndex index;

        approx_knn_build_index(handle_,
                               &index,
                               dynamic_cast<raft::spatial::knn::knnIndexParam*>(&ivfParams),
                               ps.metric,
                               (IdxT)0,
                               database.data(),
                               ps.num_db_vecs,
                               ps.dim);

        resource::sync_stream(handle_);
        approx_knn_search(handle_,
                          distances_ivfflat_dev.data(),
                          indices_ivfflat_dev.data(),
                          &index,
                          ps.k,
                          search_queries.data(),
                          ps.num_queries);

        update_host(distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
        update_host(indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
        resource::sync_stream(handle_);
      }

      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfflat,
                                  distances_naive,
                                  distances_ivfflat,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
      {
        ivf_flat::index_params index_params;
        ivf_flat::search_params search_params;
        index_params.n_lists          = ps.nlist;
        index_params.metric           = ps.metric;
        index_params.adaptive_centers = ps.adaptive_centers;
        search_params.n_probes        = ps.nprobe;

        index_params.add_data_on_build        = false;
        index_params.kmeans_trainset_fraction = 0.5;
        index_params.metric_arg               = 0;

        auto database_view = raft::make_device_matrix_view<const DataT, IdxT>(
          (const DataT*)database.data(), ps.num_db_vecs, ps.dim);

        auto idx = ivf_flat::build(handle_, index_params, database_view);

        rmm::device_uvector<IdxT> vector_indices(ps.num_db_vecs, stream_);
        thrust::sequence(resource::get_thrust_policy(handle_),
                         thrust::device_pointer_cast(vector_indices.data()),
                         thrust::device_pointer_cast(vector_indices.data() + ps.num_db_vecs));
        resource::sync_stream(handle_);

        IdxT half_of_data = ps.num_db_vecs / 2;

        auto half_of_data_view = raft::make_device_matrix_view<const DataT, IdxT>(
          (const DataT*)database.data(), half_of_data, ps.dim);

        const std::optional<raft::device_vector_view<const IdxT, IdxT>> no_opt = std::nullopt;
        index<DataT, IdxT> index_2 = ivf_flat::extend(handle_, half_of_data_view, no_opt, idx);

        auto new_half_of_data_view = raft::make_device_matrix_view<const DataT, IdxT>(
          database.data() + half_of_data * ps.dim, IdxT(ps.num_db_vecs) - half_of_data, ps.dim);

        auto new_half_of_data_indices_view = raft::make_device_vector_view<const IdxT, IdxT>(
          vector_indices.data() + half_of_data, IdxT(ps.num_db_vecs) - half_of_data);

        ivf_flat::extend(handle_,
                         new_half_of_data_view,
                         std::make_optional<raft::device_vector_view<const IdxT, IdxT>>(
                           new_half_of_data_indices_view),
                         &index_2);

        auto search_queries_view = raft::make_device_matrix_view<const DataT, IdxT>(
          search_queries.data(), ps.num_queries, ps.dim);
        auto indices_out_view = raft::make_device_matrix_view<IdxT, IdxT>(
          indices_ivfflat_dev.data(), ps.num_queries, ps.k);
        auto dists_out_view = raft::make_device_matrix_view<T, IdxT>(
          distances_ivfflat_dev.data(), ps.num_queries, ps.k);
        ivf_flat::detail::serialize(handle_, "ivf_flat_index", index_2);

        auto index_loaded = ivf_flat::detail::deserialize<DataT, IdxT>(handle_, "ivf_flat_index");
        ASSERT_EQ(index_2.size(), index_loaded.size());

        ivf_flat::search(handle_,
                         search_params,
                         index_loaded,
                         search_queries_view,
                         indices_out_view,
                         dists_out_view);

        update_host(distances_ivfflat.data(), distances_ivfflat_dev.data(), queries_size, stream_);
        update_host(indices_ivfflat.data(), indices_ivfflat_dev.data(), queries_size, stream_);
        resource::sync_stream(handle_);

        // Test the centroid invariants
        if (index_2.adaptive_centers()) {
          // The centers must be up-to-date with the corresponding data
          std::vector<uint32_t> list_sizes(index_2.n_lists());
          std::vector<IdxT*> list_indices(index_2.n_lists());
          rmm::device_uvector<float> centroid(ps.dim, stream_);
          raft::copy(
            list_sizes.data(), index_2.list_sizes().data_handle(), index_2.n_lists(), stream_);
          raft::copy(
            list_indices.data(), index_2.inds_ptrs().data_handle(), index_2.n_lists(), stream_);
          resource::sync_stream(handle_);
          for (uint32_t l = 0; l < index_2.n_lists(); l++) {
            if (list_sizes[l] == 0) continue;
            rmm::device_uvector<float> cluster_data(list_sizes[l] * ps.dim, stream_);
            raft::spatial::knn::detail::utils::copy_selected<float>((IdxT)list_sizes[l],
                                                                    (IdxT)ps.dim,
                                                                    database.data(),
                                                                    list_indices[l],
                                                                    (IdxT)ps.dim,
                                                                    cluster_data.data(),
                                                                    (IdxT)ps.dim,
                                                                    stream_);
            raft::stats::mean<float, uint32_t>(
              centroid.data(), cluster_data.data(), ps.dim, list_sizes[l], false, true, stream_);
            ASSERT_TRUE(raft::devArrMatch(index_2.centers().data_handle() + ps.dim * l,
                                          centroid.data(),
                                          ps.dim,
                                          raft::CompareApprox<float>(0.001),
                                          stream_));
          }
        } else {
          // The centers must be immutable
          ASSERT_TRUE(raft::devArrMatch(index_2.centers().data_handle(),
                                        idx.centers().data_handle(),
                                        index_2.centers().size(),
                                        raft::Compare<float>(),
                                        stream_));
        }
      }
      ASSERT_TRUE(eval_neighbours(indices_naive,
                                  indices_ivfflat,
                                  distances_naive,
                                  distances_ivfflat,
                                  ps.num_queries,
                                  ps.k,
                                  0.001,
                                  min_recall));
    }
  }

  void SetUp() override
  {
    database.resize(ps.num_db_vecs * ps.dim, stream_);
    search_queries.resize(ps.num_queries * ps.dim, stream_);

    raft::random::Rng r(1234ULL);
    if constexpr (std::is_same<DataT, float>{}) {
      r.uniform(database.data(), ps.num_db_vecs * ps.dim, DataT(0.1), DataT(2.0), stream_);
      r.uniform(search_queries.data(), ps.num_queries * ps.dim, DataT(0.1), DataT(2.0), stream_);
    } else {
      r.uniformInt(database.data(), ps.num_db_vecs * ps.dim, DataT(1), DataT(20), stream_);
      r.uniformInt(search_queries.data(), ps.num_queries * ps.dim, DataT(1), DataT(20), stream_);
    }
    resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    resource::sync_stream(handle_);
    database.resize(0, stream_);
    search_queries.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnIvfFlatInputs<IdxT> ps;
  rmm::device_uvector<DataT> database;
  rmm::device_uvector<DataT> search_queries;
};

const std::vector<AnnIvfFlatInputs<int64_t>> inputs = {
  // test various dims (aligned and not aligned to vector sizes)
  {1000, 10000, 1, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 3, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 4, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 5, 16, 40, 1024, raft::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 8, 16, 40, 1024, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 5, 16, 40, 1024, raft::distance::DistanceType::L2SqrtExpanded, false},
  {1000, 10000, 8, 16, 40, 1024, raft::distance::DistanceType::L2SqrtExpanded, true},

  // test dims that do not fit into kernel shared memory limits
  {1000, 10000, 2048, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2049, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 2050, 16, 40, 1024, raft::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 2051, 16, 40, 1024, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 2052, 16, 40, 1024, raft::distance::DistanceType::InnerProduct, false},
  {1000, 10000, 2053, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, true},
  {1000, 10000, 2056, 16, 40, 1024, raft::distance::DistanceType::L2Expanded, true},

  // various random combinations
  {1000, 10000, 16, 10, 40, 1024, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 50, 1024, raft::distance::DistanceType::L2Expanded, false},
  {1000, 10000, 16, 10, 70, 1024, raft::distance::DistanceType::L2Expanded, false},
  {100, 10000, 16, 10, 20, 512, raft::distance::DistanceType::L2Expanded, false},
  {20, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::L2Expanded, true},
  {1000, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::L2Expanded, true},
  {10000, 131072, 8, 10, 20, 1024, raft::distance::DistanceType::L2Expanded, false},

  {1000, 10000, 16, 10, 40, 1024, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 50, 1024, raft::distance::DistanceType::InnerProduct, true},
  {1000, 10000, 16, 10, 70, 1024, raft::distance::DistanceType::InnerProduct, false},
  {100, 10000, 16, 10, 20, 512, raft::distance::DistanceType::InnerProduct, true},
  {20, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::InnerProduct, true},
  {1000, 100000, 16, 10, 20, 1024, raft::distance::DistanceType::InnerProduct, false},
  {10000, 131072, 8, 10, 50, 1024, raft::distance::DistanceType::InnerProduct, true},

  {1000, 10000, 4096, 20, 50, 1024, raft::distance::DistanceType::InnerProduct, false},

  // test splitting the big query batches  (> max gridDim.y) into smaller batches
  {100000, 1024, 32, 10, 64, 64, raft::distance::DistanceType::InnerProduct, false},
  {1000000, 1024, 32, 10, 256, 256, raft::distance::DistanceType::InnerProduct, false},
  {98306, 1024, 32, 10, 64, 64, raft::distance::DistanceType::InnerProduct, true},

  // test radix_sort for getting the cluster selection
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 2,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::distance::DistanceType::L2Expanded,
   false},
  {1000,
   10000,
   16,
   10,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::matrix::detail::select::warpsort::kMaxCapacity * 4,
   raft::distance::DistanceType::InnerProduct,
   false}};

}  // namespace raft::neighbors::ivf_flat
