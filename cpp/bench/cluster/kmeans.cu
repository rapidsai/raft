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

#include <common/benchmark.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>

#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif

namespace raft::bench::cluster {

struct KMeansBenchParams {
  DatasetParams data;
  BlobsParams blobs;
  raft::cluster::KMeansParams kmeans;
};

inline auto operator<<(std::ostream& os, const KMeansBenchParams& p) -> std::ostream&
{
  os << p.data.rows << "#" << p.data.cols << "#" << p.kmeans.n_clusters;
  return os;
}

template <typename T, typename IndexT = int>
struct KMeans : public BlobsFixture<T, IndexT> {
  KMeans(const KMeansBenchParams& p) : BlobsFixture<T, IndexT>(p.data, p.blobs), params(p) {}

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    raft::device_matrix_view<const T, IndexT> X_view                          = this->X.view();
    std::optional<raft::device_vector_view<const T, IndexT>> opt_weights_view = std::nullopt;
    std::optional<raft::device_matrix_view<T, IndexT>> centroids_view =
      std::make_optional<raft::device_matrix_view<T, IndexT>>(centroids.view());
    raft::device_vector_view<IndexT, IndexT> labels_view = labels.view();
    raft::host_scalar_view<T> inertia_view               = raft::make_host_scalar_view<T>(&inertia);
    raft::host_scalar_view<IndexT> n_iter_view = raft::make_host_scalar_view<IndexT>(&n_iter);

    this->loop_on_state(state, [&]() {
      raft::cluster::kmeans_fit_predict<T, IndexT>(this->handle,
                                                   params.kmeans,
                                                   X_view,
                                                   opt_weights_view,
                                                   centroids_view,
                                                   labels_view,
                                                   inertia_view,
                                                   n_iter_view);
    });
  }

  void allocate_temp_buffers(const ::benchmark::State& state) override
  {
    centroids =
      raft::make_device_matrix<T, IndexT>(this->handle, params.kmeans.n_clusters, params.data.cols);
    labels = raft::make_device_vector<IndexT, IndexT>(this->handle, params.data.rows);
  }

 private:
  KMeansBenchParams params;
  raft::device_matrix<T, IndexT> centroids;
  raft::device_vector<IndexT, IndexT> labels;
  T inertia;
  IndexT n_iter;
};  // struct KMeans

std::vector<KMeansBenchParams> getKMeansInputs()
{
  std::vector<KMeansBenchParams> out;
  KMeansBenchParams p;
  p.data.row_major                                  = true;
  p.blobs.cluster_std                               = 1.0;
  p.blobs.shuffle                                   = false;
  p.blobs.center_box_min                            = -10.0;
  p.blobs.center_box_max                            = 10.0;
  p.blobs.seed                                      = 12345ULL;
  p.kmeans.init                                     = raft::cluster::KMeansParams::KMeansPlusPlus;
  p.kmeans.max_iter                                 = 300;
  p.kmeans.tol                                      = 1e-4;
  p.kmeans.verbosity                                = RAFT_LEVEL_INFO;
  p.kmeans.metric                                   = raft::distance::DistanceType::L2Expanded;
  p.kmeans.inertia_check                            = true;
  std::vector<std::tuple<int, int, int>> row_cols_k = {
    {1000000, 20, 1000},
    {3000000, 50, 20},
    {10000000, 50, 5},
  };
  for (auto& rck : row_cols_k) {
    p.data.rows         = std::get<0>(rck);
    p.data.cols         = std::get<1>(rck);
    p.blobs.n_clusters  = std::get<2>(rck);
    p.kmeans.n_clusters = std::get<2>(rck);
    out.push_back(p);
  }
  return out;
}

// note(lsugy): commenting out int64_t because the templates are not compiled in the distance
// library, resulting in long compilation times.
RAFT_BENCH_REGISTER((KMeans<float, int>), "", getKMeansInputs());
RAFT_BENCH_REGISTER((KMeans<double, int>), "", getKMeansInputs());
// RAFT_BENCH_REGISTER((KMeans<float, int64_t>), "", getKMeansInputs());
// RAFT_BENCH_REGISTER((KMeans<double, int64_t>), "", getKMeansInputs());

}  // namespace raft::bench::cluster
