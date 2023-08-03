#include <algorithm>
#include <memory>
#include <optional>
#include <raft/core/device_resources.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/optional.h>
namespace raft {

struct device_setter {
  static auto get_current_device() {
    auto result = int{};
    RAFT_CUDA_TRY(cudaGetDevice(&result));
    return result;
  }

  explicit device_setter(int new_device) : prev_device_{get_current_device()} {
    RAFT_CUDA_TRY(cudaSetDevice(new_device));
  }
  ~device_setter() {
    RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
  }

 private:
    int prev_device_;
};

struct resource_manager {
  resource_manager(resource_manager const&) = delete;
  void operator=(resource_manager const&) = delete;

 private:
  resource_manager() {}

  static auto get_thread_id() {
    static std::atomic<std::size_t> thread_counter{};
    thread_local std::size_t id = ++thread_counter;
    return id;
  }

  struct resource_params {
    std::optional<std::size_t> stream_count{std::nullopt};
    std::size_t pool_count{};
    std::size_t pool_size{};
    thrust::optional<std::size_t> init_mem_pool_size{std::size_t{}};
    thrust::optional<std::size_t> max_mem_pool_size{std::size_t{}};
    std::optional<std::size_t> workspace_allocation_limit{std::nullopt};
  } params_;

  struct resource_components {
    resource_components(int device_id, resource_params const& params) :
      device_id_{device_id},
      streams_{[&params, this]() {
        auto scoped_device = device_setter{device_id_};
        auto result = std::unique_ptr<rmm::cuda_stream_pool>{nullptr};
        if(params.stream_count) {
          result = std::make_unique<rmm::cuda_stream_pool>(
            *params.stream_count
          );
        }
        return result;
      }()},
      pools_{[&params, this]() {
        auto scoped_device = device_setter{device_id_};
        auto result = std::vector<std::shared_ptr<rmm::cuda_stream_pool>>{};
        if(params.pool_size != 0) {
          for (auto i = std::size_t{}; i < params.pool_count; ++i){
            result.push_back(std::make_shared<rmm::cuda_stream_pool>(
              params.pool_size
            ));
          }
        } else if (params.pool_count != 0) {
          RAFT_LOG_WARN("Stream pools of size 0 requested; no pools will be created");
        }
        return result;
      }()},
      pool_mr_{[&params, this]() {
        auto scoped_device = device_setter{device_id_};
        auto result = std::shared_ptr<
          rmm::mr::pool_memory_resource<
            rmm::mr::cuda_memory_resource
          >
        >{nullptr};
        // If max_mem_pool_size is nullopt or non-zero, create a pool memory
        // resource
        if (params.max_mem_pool_size.value_or(1) != 0) {
          auto* upstream = dynamic_cast<rmm::mr::cuda_memory_resource*>(
            rmm::mr::get_current_device_resource()
          );
          if (upstream != nullptr) {
            result = std::make_shared<rmm::mr::pool_memory_resource<
              rmm::mr::cuda_memory_resource
            >>(
              upstream,
              params.init_mem_pool_size,
              params.max_mem_pool_size
            );
          } else {
            RAFT_LOG_WARN(
              "Pool allocation requested, but other memory resource has already been set and will not be overwritten"
            );
          }
        }
        return result;
      }()} {}
    [[nodiscard]] auto get_device_id() const {
      return device_id_;
    }
    [[nodiscard]] auto stream_count() const {
      return streams_->get_pool_size();
    }
    [[nodiscard]] auto get_stream() const {
      auto result = rmm::cuda_stream_per_thread;
      if(stream_count() != 0) {
        result = streams_->get_stream(
          get_thread_id() % stream_count()
        );
      }
      return result;
    }
    [[nodiscard]] auto pool_count() const {
      return pools_.size();
    }
    [[nodiscard]] auto get_pool() const {
      auto result = std::shared_ptr<rmm::cuda_stream_pool>{nullptr};
      if (pool_count() != 0) {
        result = pools_[get_thread_id() % pool_count()];
      }
      return result;
    }
    [[nodiscard]] auto get_pool_memory_resource() const {
      return pool_mr_;
    }
    [[nodiscard]] auto get_workspace_allocation_limit() const {
      return workspace_allocation_limit_;
    }
   private:
    int device_id_;
    std::unique_ptr<rmm::cuda_stream_pool> streams_;
    std::vector<std::shared_ptr<rmm::cuda_stream_pool>> pools_;
    std::shared_ptr<
      rmm::mr::pool_memory_resource<
        rmm::mr::cuda_memory_resource
      >
    > pool_mr_;
    std::optional<std::size_t> workspace_allocation_limit_{std::nullopt};
  };

  std::mutex manager_mutex_{};
  bool params_finalized_{};
  std::vector<resource_components> per_device_components_;

  [[nodiscard]] auto get_lock() const {
    return std::unique_lock{manager_mutex_};
  }

  auto const& get_device_components(int device_id) {
    // Each thread maintains an independent list of devices it has
    // accessed. If it has not marked a device as initialized, it
    // acquires a lock to initialize it exactly once. This means that each
    // thread will lock once for a particular device and not proceed until
    // some thread has actually generated the corresponding device
    // components
    thread_local auto initialized_devices = std::vector<int>{};
    auto iter = std::end(per_device_components_);
    if (
      std::find(
        std::begin(initialized_devices),
        std::end(initialized_devices),
        device_id
      ) == std::end(initialized_devices)
    ) {
      // Only lock if we have not previously accessed this device on this
      // thread
      auto lock = get_lock();
      initialized_devices.push_back(device_id);
      // If we are building components, do not allow any further changes to
      // resource parameters.
      params_finalized_ = true;

      iter = std::find_if(
        std::begin(per_device_components_),
        std::end(per_device_components_),
        [device_id](auto&& components) {
          return components.get_device_id() == device_id;
        }
      );
      if (iter == per_device_components_.end()) {
        per_device_components_.emplace_back(device_id, params_);
        iter = std::prev(std::end(per_device_components_));
      }
    } else {
      // If we have previously accessed this device on this thread, we do not
      // need to lock. We know that this thread already initialized the device
      // if no other thread had already done so, so we simply retrieve the
      // components for this device.
      iter = std::find_if(
        std::begin(per_device_components_),
        std::end(per_device_components_),
        [device_id](auto&& components) {
          return components.get_device_id() == device_id;
        }
      );
    }
    return *iter;
  }

  void set_streams_per_device_(std::optional<std::size_t> num_streams) {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN("Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      params_.stream_count = num_streams;
    }
  }

  void set_stream_pools_per_device_(std::size_t num_pools, std::size_t num_streams) {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN("Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      params_.pool_count = num_pools;
      params_.pool_size = num_streams;
    }
  }

  void set_workspace_allocation_limit_(std::size_t memory_limit) {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN("Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      params_.workspace_allocation_limit.emplace(memory_limit);
    }
  }

  void set_max_mem_pool_size_(std::optional<std::size_t> memory_limit) {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN("Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      if (memory_limit) {
        params_.max_mem_pool_size.emplace(*memory_limit);
      } else {
        params_.max_mem_pool_size = thrust::nullopt;
      }
    }
  }

  void set_init_mem_pool_size_(std::optional<std::size_t> init_memory) {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN("Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      if (init_memory) {
        params_.init_mem_pool_size.emplace(*init_memory);
      } else {
        params_.init_mem_pool_size = thrust::nullopt;
      }
    }
  }

  static auto& get_manager() {
    static auto manager = resource_manager{};
    return manager;
  }

 public:
  static auto get_resources(int device_id=device_setter::get_current_device(), std::shared_ptr<rmm::mr::device_memory_resource> workspace_mr = {nullptr}) {
    auto const& components = get_manager().get_device_components(device_id);
    return device_resources{
      components.get_stream(),
      components.get_pool(),
      workspace_mr ? workspace_mr : components.get_pool_memory_resource(),
      components.get_workspace_allocation_limit()
    };
  }

  static void set_streams_per_device(std::optional<std::size_t> num_streams) {
    get_manager().set_streams_per_device_(num_streams);
  }
  static void set_stream_pools_per_device(std::size_t num_pools, std::size_t num_streams=rmm::cuda_stream_pool::default_size) {
    get_manager().set_stream_pools_per_device_(num_pools, num_streams);
  }
  static void set_workspace_allocation_limit(std::size_t memory_limit) {
    get_manager().set_workspace_allocation_limit_(memory_limit);
  }

  static void set_max_mem_pool_size(std::optional<std::size_t> max_mem) {
    get_manager().set_max_mem_pool_size_(max_mem);
  }
  static void set_init_mem_pool_size(std::optional<std::size_t> init_mem) {
    get_manager().set_init_mem_pool_size_(init_mem);
  }
  static void set_mem_pool(std::optional<std::size_t> init_mem=std::nullopt, std::optional<std::size_t> max_mem=std::nullopt) {
    set_init_mem_pool_size(init_mem);
    set_max_mem_pool_size(max_mem);
  }
};
}  // namespace raft
