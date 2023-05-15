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

#include <common/benchmark.hpp>

#include <raft/random/rng.cuh>

#include <raft/neighbors/ivf_flat.cuh>
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/spatial/knn/knn.cuh>

#if defined RAFT_COMPILED
#include <raft/neighbors/specializations.cuh>
#endif

#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <optional>

namespace raft::bench::spatial {

struct params {
  /** Size of the dataset. */
  size_t n_samples;
  /** Number of dimensions in the dataset. */
  size_t n_dims;
  /** The batch size -- number of KNN searches. */
  size_t n_queries;
  /** Number of nearest neighbours to find for every probe. */
  size_t k;
};

inline auto operator<<(std::ostream& os, const params& p) -> std::ostream&
{
  os << p.n_samples << "#" << p.n_dims << "#" << p.n_queries << "#" << p.k;
  return os;
}

enum class TransferStrategy { NO_COPY, COPY_PLAIN, COPY_PINNED, MAP_PINNED, MANAGED };  // NOLINT
enum class Scope { BUILD, SEARCH, BUILD_SEARCH };                                       // NOLINT

inline auto operator<<(std::ostream& os, const TransferStrategy& ts) -> std::ostream&
{
  switch (ts) {
    case TransferStrategy::NO_COPY: os << "NO_COPY"; break;
    case TransferStrategy::COPY_PLAIN: os << "COPY_PLAIN"; break;
    case TransferStrategy::COPY_PINNED: os << "COPY_PINNED"; break;
    case TransferStrategy::MAP_PINNED: os << "MAP_PINNED"; break;
    case TransferStrategy::MANAGED: os << "MANAGED"; break;
    default: os << "UNKNOWN";
  }
  return os;
}

inline auto operator<<(std::ostream& os, const Scope& s) -> std::ostream&
{
  switch (s) {
    case Scope::BUILD: os << "BUILD"; break;
    case Scope::SEARCH: os << "SEARCH"; break;
    case Scope::BUILD_SEARCH: os << "BUILD_SEARCH"; break;
    default: os << "UNKNOWN";
  }
  return os;
}

struct device_resource {
 public:
  explicit device_resource(bool managed) : managed_(managed)
  {
    if (managed_) {
      res_ = new rmm::mr::managed_memory_resource();
    } else {
      res_ = rmm::mr::get_current_device_resource();
    }
  }

  ~device_resource()
  {
    if (managed_) { delete res_; }
  }

  [[nodiscard]] auto get() const -> rmm::mr::device_memory_resource* { return res_; }

 private:
  const bool managed_;
  rmm::mr::device_memory_resource* res_;
};

template <typename T>
struct host_uvector {
  host_uvector(size_t n, bool pinned) : n_(n)
  {
    if (pinned) {
      res_ = new rmm::mr::pinned_memory_resource();
    } else {
      res_ = new rmm::mr::new_delete_resource();
    }
    arr_ = static_cast<T*>(res_->allocate(n_ * sizeof(T)));
  }

  ~host_uvector() noexcept
  {
    res_->deallocate(arr_, n_ * sizeof(T));
    delete res_;
  }

  auto data() -> T* { return arr_; }
  [[nodiscard]] auto size() const -> size_t { return n_; }

 private:
  rmm::mr::host_memory_resource* res_;
  size_t n_;
  T* arr_;
};

template <typename ValT, typename IdxT>
struct ivf_flat_knn {
  using dist_t = float;

  std::optional<const raft::neighbors::ivf_flat::index<ValT, IdxT>> index;
  raft::neighbors::ivf_flat::index_params index_params;
  raft::neighbors::ivf_flat::search_params search_params;
  params ps;

  ivf_flat_knn(const raft::device_resources& handle, const params& ps, const ValT* data) : ps(ps)
  {
    index_params.n_lists = 4096;
    index_params.metric  = raft::distance::DistanceType::L2Expanded;
    index.emplace(raft::neighbors::ivf_flat::build(
      handle, index_params, data, IdxT(ps.n_samples), uint32_t(ps.n_dims)));
  }

  void search(const raft::device_resources& handle,
              const ValT* search_items,
              dist_t* out_dists,
              IdxT* out_idxs)
  {
    search_params.n_probes = 20;
    raft::neighbors::ivf_flat::search(
      handle, search_params, *index, search_items, ps.n_queries, ps.k, out_idxs, out_dists);
  }
};

template <typename ValT, typename IdxT>
struct ivf_pq_knn {
  using dist_t = float;

  std::optional<const raft::neighbors::ivf_pq::index<IdxT>> index;
  raft::neighbors::ivf_pq::index_params index_params;
  raft::neighbors::ivf_pq::search_params search_params;
  params ps;

  ivf_pq_knn(const raft::device_resources& handle, const params& ps, const ValT* data) : ps(ps)
  {
    index_params.n_lists = 4096;
    index_params.metric  = raft::distance::DistanceType::L2Expanded;
    auto data_view = raft::make_device_matrix_view<const ValT, IdxT>(data, ps.n_samples, ps.n_dims);
    index.emplace(raft::neighbors::ivf_pq::build(handle, index_params, data_view));
  }

  void search(const raft::device_resources& handle,
              const ValT* search_items,
              dist_t* out_dists,
              IdxT* out_idxs)
  {
    search_params.n_probes = 20;
    auto queries_view =
      raft::make_device_matrix_view<const ValT, IdxT>(search_items, ps.n_queries, ps.n_dims);
    auto idxs_view  = raft::make_device_matrix_view<IdxT, IdxT>(out_idxs, ps.n_queries, ps.k);
    auto dists_view = raft::make_device_matrix_view<dist_t, IdxT>(out_dists, ps.n_queries, ps.k);
    raft::neighbors::ivf_pq::search(
      handle, search_params, *index, queries_view, idxs_view, dists_view);
  }
};

template <typename ValT, typename IdxT>
struct brute_force_knn {
  using dist_t = ValT;

  ValT* index;
  params ps;

  brute_force_knn(const raft::device_resources& handle, const params& ps, const ValT* data)
    : index(const_cast<ValT*>(data)), ps(ps)
  {
  }

  void search(const raft::device_resources& handle,
              const ValT* search_items,
              dist_t* out_dists,
              IdxT* out_idxs)
  {
    std::vector<ValT*> input{index};
    std::vector<size_t> sizes{ps.n_samples};
    raft::spatial::knn::brute_force_knn<IdxT, ValT, size_t>(handle,
                                                            input,
                                                            sizes,
                                                            ps.n_dims,
                                                            const_cast<ValT*>(search_items),
                                                            ps.n_queries,
                                                            out_idxs,
                                                            out_dists,
                                                            ps.k);
  }
};

template <typename ValT, typename IdxT, typename ImplT>
struct knn : public fixture {
  explicit knn(const params& p, const TransferStrategy& strategy, const Scope& scope)
    : params_(p),
      strategy_(strategy),
      scope_(scope),
      dev_mem_res_(strategy == TransferStrategy::MANAGED),
      data_host_(0),
      search_items_(p.n_queries * p.n_dims, stream),
      out_dists_(p.n_queries * p.k, stream),
      out_idxs_(p.n_queries * p.k, stream)
  {
    raft::random::RngState state{42};
    gen_data(state, search_items_, search_items_.size(), stream);
    try {
      size_t total_size = p.n_samples * p.n_dims;
      data_host_.resize(total_size);
      constexpr size_t kGenMinibatchSize = 1024 * 1024 * 1024;
      rmm::device_uvector<ValT> d(std::min(kGenMinibatchSize, total_size), stream);
      for (size_t offset = 0; offset < total_size; offset += kGenMinibatchSize) {
        size_t actual_size = std::min(total_size - offset, kGenMinibatchSize);
        gen_data(state, d, actual_size, stream);
        copy(data_host_.data() + offset, d.data(), actual_size, stream);
      }
    } catch (std::bad_alloc& e) {
      data_does_not_fit_ = true;
    }
  }

  template <typename T>
  void gen_data(raft::random::RngState& state,  // NOLINT
                rmm::device_uvector<T>& vec,
                size_t n,
                rmm::cuda_stream_view stream)
  {
    constexpr T kRangeMax = std::is_integral_v<T> ? std::numeric_limits<T>::max() : T(1);
    constexpr T kRangeMin = std::is_integral_v<T> ? std::numeric_limits<T>::min() : T(-1);
    if constexpr (std::is_integral_v<T>) {
      raft::random::uniformInt(state, vec.data(), n, kRangeMin, kRangeMax, stream);
    } else {
      raft::random::uniform(state, vec.data(), n, kRangeMin, kRangeMax, stream);
    }
  }

  void run_benchmark(::benchmark::State& state) override
  {
    if (data_does_not_fit_) {
      state.SkipWithError("The data size is too big to fit into the host memory.");
    }
    if (scope_ == Scope::SEARCH && strategy_ != TransferStrategy::NO_COPY) {
      state.SkipWithError(
        "When benchmarking without index building (Scope::SEARCH), the data must be already on the "
        "device (TransferStrategy::NO_COPY)");
    }

    using_pool_memory_res default_resource;

    try {
      std::ostringstream label_stream;
      label_stream << params_ << "#" << strategy_ << "#" << scope_;
      state.SetLabel(label_stream.str());
      raft::device_resources handle(stream);
      std::optional<ImplT> index;

      if (scope_ == Scope::SEARCH) {  // also implies TransferStrategy::NO_COPY
        rmm::device_uvector<ValT> data(data_host_.size(), stream);
        copy(data.data(), data_host_.data(), data_host_.size(), stream);
        index.emplace(handle, params_, data.data());
        stream.synchronize();
      }

      // benchmark loop
      for (auto _ : state) {
        // managed or plain device memory initialized anew every time
        rmm::device_uvector<ValT> data(data_host_.size(), stream, dev_mem_res_.get());
        ValT* data_ptr         = data.data();
        size_t allocation_size = data_host_.size() * sizeof(ValT);

        // Non-benchmarked part: using different methods to copy the data if necessary
        switch (strategy_) {
          case TransferStrategy::NO_COPY:  // copy data to GPU before starting the timer.
            copy(data_ptr, data_host_.data(), data_host_.size(), stream);
            break;
          case TransferStrategy::COPY_PINNED:
            RAFT_CUDA_TRY(
              cudaHostRegister(data_host_.data(), allocation_size, cudaHostRegisterDefault));
            break;
          case TransferStrategy::MAP_PINNED:
            RAFT_CUDA_TRY(
              cudaHostRegister(data_host_.data(), allocation_size, cudaHostRegisterMapped));
            RAFT_CUDA_TRY(cudaHostGetDevicePointer(&data_ptr, data_host_.data(), 0));
            break;
          case TransferStrategy::MANAGED:  // sic! using std::memcpy rather than cuda copy
            RAFT_CUDA_TRY(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetPreferredLocation, handle.get_device()));
            RAFT_CUDA_TRY(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetAccessedBy, handle.get_device()));
            RAFT_CUDA_TRY(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetReadMostly, handle.get_device()));
            std::memcpy(data_ptr, data_host_.data(), allocation_size);
            break;
          default: break;
        }

        flush_L2_cache();
        {
          // Timer synchronizes the stream, so all prior gpu work should be done before it sets off.
          cuda_event_timer timer(state, stream);
          switch (strategy_) {
            case TransferStrategy::COPY_PLAIN:
            case TransferStrategy::COPY_PINNED:
              copy(data_ptr, data_host_.data(), data_host_.size(), stream);
            default: break;
          }

          if (scope_ != Scope::SEARCH) { index.emplace(handle, params_, data_ptr); }
          if (scope_ != Scope::BUILD) {
            index->search(handle, search_items_.data(), out_dists_.data(), out_idxs_.data());
          }
        }

        if (scope_ != Scope::SEARCH) { index.reset(); }

        switch (strategy_) {
          case TransferStrategy::COPY_PINNED:
          case TransferStrategy::MAP_PINNED:
            RAFT_CUDA_TRY(cudaHostUnregister(data_host_.data()));
            break;
          default: break;
        }
      }
    } catch (raft::exception& e) {
      state.SkipWithError(e.what());
    } catch (std::bad_alloc& e) {
      state.SkipWithError(e.what());
    }
  }

 private:
  const params params_;
  const TransferStrategy strategy_;
  const Scope scope_;
  device_resource dev_mem_res_;
  bool data_does_not_fit_ = false;

  std::vector<ValT> data_host_;
  rmm::device_uvector<ValT> search_items_;
  rmm::device_uvector<typename ImplT::dist_t> out_dists_;
  rmm::device_uvector<IdxT> out_idxs_;
};

inline const std::vector<params> kInputs{
  {2000000, 128, 1000, 32}, {10000000, 128, 1000, 32}, {10000, 8192, 1000, 32}};

inline const std::vector<TransferStrategy> kAllStrategies{
  TransferStrategy::NO_COPY, TransferStrategy::MAP_PINNED, TransferStrategy::MANAGED};
inline const std::vector<TransferStrategy> kNoCopyOnly{TransferStrategy::NO_COPY};

inline const std::vector<Scope> kScopeFull{Scope::BUILD_SEARCH};
inline const std::vector<Scope> kAllScopes{Scope::BUILD_SEARCH, Scope::SEARCH, Scope::BUILD};

#define KNN_REGISTER(ValT, IdxT, ImplT, inputs, strats, scope)                   \
  namespace BENCHMARK_PRIVATE_NAME(knn)                                          \
  {                                                                              \
    using KNN = knn<ValT, IdxT, ImplT<ValT, IdxT>>;                              \
    RAFT_BENCH_REGISTER(KNN, #ValT "/" #IdxT "/" #ImplT, inputs, strats, scope); \
  }

}  // namespace raft::bench::spatial
