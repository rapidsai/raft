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

#include <optional>

#include <common/benchmark.hpp>
#include <raft/spatial/knn/ann.cuh>
#include <raft/spatial/knn/knn.cuh>

#if defined RAFT_NN_COMPILED
#include <raft/spatial/knn/specializations.hpp>
#endif

#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>

#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <rmm/mr/host/new_delete_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

namespace raft::bench::spatial {

struct params {
  /** Size of the dataset. */
  size_t n_samples;
  /** Number of dimensions in the dataset. */
  size_t n_dims;
  /** The batch size -- number of KNN searches. */
  size_t n_probes;
  /** Number of nearest neighbours to find for every probe. */
  size_t k;
};

auto operator<<(std::ostream& os, const params& p) -> std::ostream&
{
  os << p.n_samples << "#" << p.n_dims << "#" << p.n_probes << "#" << p.k;
  return os;
}

enum class TransferStrategy { NO_COPY, COPY_PLAIN, COPY_PINNED, MAP_PINNED, MANAGED };
enum class Scope { BUILD, SEARCH, BUILD_SEARCH };

auto operator<<(std::ostream& os, const TransferStrategy& ts) -> std::ostream&
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

auto operator<<(std::ostream& os, const Scope& s) -> std::ostream&
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
  raft::spatial::knn::knnIndex index;
  raft::spatial::knn::ivf_flat::index_params index_params;
  raft::spatial::knn::ivf_flat::search_params search_params;
  params ps;

  ivf_flat_knn(const raft::handle_t& handle, const params& ps, const ValT* data) : ps(ps)
  {
    index_params.n_lists = 4096;
    index_params.metric  = raft::distance::DistanceType::L2Expanded;
    raft::spatial::knn::approx_knn_build_index<ValT, IdxT>(const_cast<raft::handle_t&>(handle),
                                                           &index,
                                                           index_params,
                                                           const_cast<ValT*>(data),
                                                           (IdxT)ps.n_samples,
                                                           (IdxT)ps.n_dims);
  }

  void search(const raft::handle_t& handle,
              const ValT* search_items,
              ValT* out_dists,
              IdxT* out_idxs)
  {
    search_params.n_probes = 20;
    raft::spatial::knn::approx_knn_search<ValT, IdxT>(const_cast<raft::handle_t&>(handle),
                                                      out_dists,
                                                      out_idxs,
                                                      &index,
                                                      search_params,
                                                      (IdxT)ps.k,
                                                      const_cast<ValT*>(search_items),
                                                      (IdxT)ps.n_probes);
  }
};

template <typename ValT, typename IdxT>
struct brute_force_knn {
  ValT* index;
  params ps;

  brute_force_knn(const raft::handle_t& handle, const params& ps, const ValT* data)
    : index(const_cast<ValT*>(data)), ps(ps)
  {
  }

  void search(const raft::handle_t& handle,
              const ValT* search_items,
              ValT* out_dists,
              IdxT* out_idxs)
  {
    std::vector<ValT*> input{index};
    std::vector<size_t> sizes{ps.n_samples};
    raft::spatial::knn::brute_force_knn<IdxT, ValT, size_t>(handle,
                                                            input,
                                                            sizes,
                                                            ps.n_dims,
                                                            const_cast<ValT*>(search_items),
                                                            ps.n_probes,
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
      search_items_(p.n_probes * p.n_dims, stream),
      out_dists_(p.n_probes * p.k, stream),
      out_idxs_(p.n_probes * p.k, stream)
  {
    raft::random::RngState state{42};
    raft::random::uniform(
      state, search_items_.data(), search_items_.size(), ValT(-1.0), ValT(1.0), stream);
    try {
      size_t total_size = p.n_samples * p.n_dims;
      data_host_.resize(total_size);
      constexpr size_t kGenMinibatchSize = 1024 * 1024 * 1024;
      rmm::device_uvector<ValT> d(std::min(kGenMinibatchSize, total_size), stream);
      for (size_t offset = 0; offset < total_size; offset += kGenMinibatchSize) {
        size_t actual_size = std::min(total_size - offset, kGenMinibatchSize);
        raft::random::uniform(state, d.data(), actual_size, ValT(-1.0), ValT(1.0), stream);
        copy(data_host_.data() + offset, d.data(), actual_size, stream);
      }
    } catch (std::bad_alloc& e) {
      data_does_not_fit_ = true;
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
      raft::handle_t handle(stream);
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
            CUDA_CHECK(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetPreferredLocation, handle.get_device()));
            CUDA_CHECK(cudaMemAdvise(
              data_ptr, allocation_size, cudaMemAdviseSetAccessedBy, handle.get_device()));
            CUDA_CHECK(cudaMemAdvise(data_ptr, allocation_size, cudaMemAdviseSetReadMostly, 0));
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
  rmm::device_uvector<ValT> out_dists_;
  rmm::device_uvector<IdxT> out_idxs_;
};

const std::vector<params> kInputs{
  {2000000, 128, 1000, 32}, {10000000, 128, 1000, 32}, {10000, 8192, 1000, 32}};

const std::vector<TransferStrategy> kAllStrategies{TransferStrategy::NO_COPY,
                                                   TransferStrategy::COPY_PLAIN,
                                                   TransferStrategy::COPY_PINNED,
                                                   TransferStrategy::MAP_PINNED,
                                                   TransferStrategy::MANAGED};
const std::vector<TransferStrategy> kNoCopyOnly{TransferStrategy::NO_COPY};

const std::vector<Scope> kScopeFull{Scope::BUILD_SEARCH};
const std::vector<Scope> kAllScopes{Scope::BUILD, Scope::SEARCH, Scope::BUILD_SEARCH};

#define KNN_REGISTER(ValT, IdxT, ImplT, inputs, strats, scope)                   \
  namespace BENCHMARK_PRIVATE_NAME(knn)                                          \
  {                                                                              \
    using KNN = knn<ValT, IdxT, ImplT<ValT, IdxT>>;                              \
    RAFT_BENCH_REGISTER(KNN, #ValT "/" #IdxT "/" #ImplT, inputs, strats, scope); \
  }

KNN_REGISTER(float, int64_t, brute_force_knn, kInputs, kAllStrategies, kScopeFull);
KNN_REGISTER(float, int64_t, ivf_flat_knn, kInputs, kNoCopyOnly, kAllScopes);

}  // namespace raft::bench::spatial
