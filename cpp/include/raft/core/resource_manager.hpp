#include <algorithm>
#include <memory>
#include <optional>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_setter.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <thrust/optional.h>
namespace raft {

/**
 * @brief A singleton used to easily generate a raft::device_resources object
 *
 * Many calls to RAFT functions require a `raft::device_resources` object
 * to provide CUDA resources like streams and stream pools. The
 * `raft::resource_manager` singleton provides a straightforward method to create those objects in
 * a way that allows consumers of RAFT to limit total consumption of device
 * resources without actively managing streams or other CUDA-specific
 * objects.
 *
 * To control the resources a consuming application will use, the
 * resource manager provides setters for a variety of values. For
 * instance, to ensure that no more than `N` CUDA streams are used per
 * device, a consumer might call
 * `raft::resource_manager::set_streams_per_device(N)`. Note that all of these
 * setters must be used prior to retrieving the first `device_resources` from
 * the manager. Setters invoked after this will log a warning but have no
 * effect.
 *
 * After calling all desired setters, consumers can simply call
 * `auto res = raft::resource_manager::get_resources();` to get a valid
 * device_resources object for the current device based on previously-set
 * parameters. Importantly, calling `get_resources()` again from the same
 * thread is guaranteed to return a `device_resources` object with the same
 * underlying CUDA stream and (if a non-zero number of stream pools has been
 * requested) stream pool.
 *
 * Typical usage might look something like the following:
 * @code
 * void initialize_application() {
 *   raft::resource_manager::set_streams_per_device(16);
 * }
 *
 * void foo_called_from_multiple_threads() {
 *   auto res = raft::resource_manager::get_device_resources();
 *   // Call RAFT function using res
 *   raft::resource_manager::synchronize_work_from_this_thread();
 * }
 * @endcode
 *
 * Note that all public methods of the `resource_manager` are thread-safe,
 * but the manager is designed to minimize locking required for
 * retrieving `device_resources` objects. Each thread must acquire a lock
 * exactly once per device when calling `get_resources`. Subsequent calls
 * will still be thread-safe but will not require a lock.
 *
 * All public methods of the `resource_manager` are static. Please see
 * documentation of those methods for additional usage information.
 *
 */
struct resource_manager {
  resource_manager(resource_manager const&) = delete;
  void operator=(resource_manager const&)   = delete;

 private:
  resource_manager() {}

  // Get an id used to identify this thread for the purposes of assigning
  // (in round-robin fashion) the same resources to the thread on subsequent calls to
  // `get_device_resources`
  static auto get_thread_id()
  {
    static std::atomic<std::size_t> thread_counter{};
    thread_local std::size_t id = ++thread_counter;
    return id;
  }

  // This struct holds the various parameters used to control
  // construction of the underlying resources shared by all
  // `device_resources` objects returned by `get_device_resources`
  struct resource_params {
    // The total number of primary streams to be used by the
    // application. If no value is provided, the default stream per thread
    // is used.
    std::optional<std::size_t> stream_count{std::nullopt};
    // The total number of stream pools to be used by the application
    std::size_t pool_count{};
    // How many streams to assign to each pool
    std::size_t pool_size{rmm::cuda_stream_pool::default_size};
    // If a memory pool is requested (max_mem_pool_size is non-zero), use
    // this initial size for the pool in bytes. Must be a multiple of 256.
    // If nullopt, use half of the available memory on the current
    // device.
    thrust::optional<std::size_t> init_mem_pool_size{thrust::nullopt};
    // If set to any non-zero value, create a memory pool with this
    // maximum size. If nullopt, use up to the entire available memory of the
    // device
    thrust::optional<std::size_t> max_mem_pool_size{std::size_t{}};
    // Limit on workspace memory for the returned device_resources object
    std::optional<std::size_t> workspace_allocation_limit{std::nullopt};
  } params_;

  // This struct stores the underlying resources to be shared among
  // `device_resources` objects returned by this manager.
  struct resource_components {
    // Construct all underlying resources indicated by `params` for the
    // indicated device. This includes primary streams, stream pools, and
    // a memory pool if requested.
    resource_components(int device_id, resource_params const& params)
      : device_id_{device_id},
        streams_{[&params, this]() {
          auto scoped_device = device_setter{device_id_};
          auto result        = std::unique_ptr<rmm::cuda_stream_pool>{nullptr};
          if (params.stream_count) {
            result = std::make_unique<rmm::cuda_stream_pool>(*params.stream_count);
          }
          return result;
        }()},
        pools_{[&params, this]() {
          auto scoped_device = device_setter{device_id_};
          auto result        = std::vector<std::shared_ptr<rmm::cuda_stream_pool>>{};
          if (params.pool_size != 0) {
            for (auto i = std::size_t{}; i < params.pool_count; ++i) {
              result.push_back(std::make_shared<rmm::cuda_stream_pool>(params.pool_size));
            }
          } else if (params.pool_count != 0) {
            RAFT_LOG_WARN("Stream pools of size 0 requested; no pools will be created");
          }
          return result;
        }()},
        pool_mr_{[&params, this]() {
          auto scoped_device = device_setter{device_id_};
          auto result =
            std::shared_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>{nullptr};
          // If max_mem_pool_size is nullopt or non-zero, create a pool memory
          // resource
          if (params.max_mem_pool_size.value_or(1) != 0) {
            auto* upstream =
              dynamic_cast<rmm::mr::cuda_memory_resource*>(rmm::mr::get_current_device_resource());
            if (upstream != nullptr) {
              result =
                std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                  upstream, params.init_mem_pool_size, params.max_mem_pool_size);
            } else {
              RAFT_LOG_WARN(
                "Pool allocation requested, but other memory resource has already been set and "
                "will not be overwritten");
            }
          }
          return result;
        }()}
    {
    }

    // Get the id of the device associated with the constructed resource
    // components
    [[nodiscard]] auto get_device_id() const { return device_id_; }
    // Get the total number of streams available for this application
    [[nodiscard]] auto stream_count() const
    {
      auto result = std::size_t{};
      if (streams_) { result = streams_->get_pool_size(); }
      return result;
    }
    // Get the stream assigned to this host thread. Note that the same stream
    // may be used by multiple threads, but any given thread will always use
    // the same stream
    [[nodiscard]] auto get_stream() const
    {
      auto result = rmm::cuda_stream_per_thread;
      if (stream_count() != 0) { result = streams_->get_stream(get_thread_id() % stream_count()); }
      return result;
    }
    // Get the total number of stream pools available for this
    // application
    [[nodiscard]] auto pool_count() const { return pools_.size(); }
    // Get the stream pool assigned to this host thread. Note that the same stream pool
    // may be used by multiple threads, but any given thread will always use
    // the same stream pool
    [[nodiscard]] auto get_pool() const
    {
      auto result = std::shared_ptr<rmm::cuda_stream_pool>{nullptr};
      if (pool_count() != 0) { result = pools_[get_thread_id() % pool_count()]; }
      return result;
    }
    // Return a (possibly null) shared_ptr to the pool memory resource
    // created for this device by the manager
    [[nodiscard]] auto get_pool_memory_resource() const { return pool_mr_; }
    // Return the RAFT workspace allocation limit that will be used by
    // `device_resources` returned from this manager
    [[nodiscard]] auto get_workspace_allocation_limit() const
    {
      return workspace_allocation_limit_;
    }
    void synchronize() const
    {
      for (auto i = std::size_t{}; i < stream_count(); ++i) {
        interruptible::synchronize(streams_->get_stream(i));
      }
      for (auto i = std::size_t{}; i < pool_count(); ++i) {
        for (auto j = std::size_t{}; j < pools_[i]->get_pool_size(); ++j) {
          interruptible::synchronize(pools_[i]->get_stream(j));
        }
      }
    }

   private:
    int device_id_;
    std::unique_ptr<rmm::cuda_stream_pool> streams_;
    std::vector<std::shared_ptr<rmm::cuda_stream_pool>> pools_;
    std::shared_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_mr_;
    std::optional<std::size_t> workspace_allocation_limit_{std::nullopt};
  };

  // Mutex used to lock access to shared data until after the first
  // `get_device_resources` call in each thread
  std::mutex manager_mutex_{};
  // Indicates whether or not `get_device_resources` has been called by any
  // host thread
  bool params_finalized_{};
  // Container for underlying device resources to be re-used across host
  // threads for each device
  std::vector<resource_components> per_device_components_;

  // Return a lock for accessing shared data
  [[nodiscard]] auto get_lock() const { return std::unique_lock{manager_mutex_}; }

  // Retrieve the underlying resources to be shared across the
  // application for the indicated device. This method acquires a lock the
  // first time it is called in each thread for a specific device to ensure that the
  // underlying resources have been correctly initialized exactly once across
  // all host threads.
  auto const& get_device_components(int device_id)
  {
    // Each thread maintains an independent list of devices it has
    // accessed. If it has not marked a device as initialized, it
    // acquires a lock to initialize it exactly once. This means that each
    // thread will lock once for a particular device and not proceed until
    // some thread has actually generated the corresponding device
    // components
    thread_local auto initialized_devices = std::vector<int>{};
    auto iter                             = std::end(per_device_components_);
    if (std::find(std::begin(initialized_devices), std::end(initialized_devices), device_id) ==
        std::end(initialized_devices)) {
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
        [device_id](auto&& components) { return components.get_device_id() == device_id; });
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
        [device_id](auto&& components) { return components.get_device_id() == device_id; });
    }
    return *iter;
  }

  // Thread-safe setter for the number of streams
  void set_streams_per_device_(std::optional<std::size_t> num_streams)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      params_.stream_count = num_streams;
    }
  }

  // Thread-safe setter for the number and size of stream pools
  void set_stream_pools_per_device_(std::size_t num_pools, std::size_t num_streams)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      params_.pool_count = num_pools;
      params_.pool_size  = num_streams;
    }
  }

  // Thread-safe setter for the RAFT workspace allocation limit
  void set_workspace_allocation_limit_(std::size_t memory_limit)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      params_.workspace_allocation_limit.emplace(memory_limit);
    }
  }

  // Thread-safe setter for the maximum memory pool size
  void set_max_mem_pool_size_(std::optional<std::size_t> memory_limit)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      if (memory_limit) {
        params_.max_mem_pool_size.emplace(*memory_limit);
      } else {
        params_.max_mem_pool_size = thrust::nullopt;
      }
    }
  }

  // Thread-safe setter for the initial memory pool size
  void set_init_mem_pool_size_(std::optional<std::size_t> init_memory)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set resource_manager properties after resources have already been retrieved");
    } else {
      if (init_memory) {
        params_.init_mem_pool_size.emplace(*init_memory);
      } else {
        params_.init_mem_pool_size = thrust::nullopt;
      }
    }
  }

  // Retrieve the instance of this singleton
  static auto& get_manager()
  {
    static auto manager = resource_manager{};
    return manager;
  }

 public:
  /**
   * @brief Retrieve device_resources to be used with the RAFT API
   *
   * This thread-safe method ensures that a `device_resources` object with
   * the same underlying stream and stream pool is returned every time it is
   * called by the same host thread. This means that if `get_resources` is
   * used to provide all `device_resources` in an application, then
   * `raft::get_resources().sync_stream()` and (if a stream pool is used)
   * raft::get_resources().sync_stream_pool() are guaranteed to synchronize all
   * work previously submitted to the device by this host thread.
   *
   * If the max memory pool size set with `set_max_mem_pool_size` is non-zero,
   * the first call of this method will also create a memory pool to be used
   * for all RMM-based allocations on device.
   *
   * @param int device_id If provided, the device for which resources should
   * be returned. Defaults to active CUDA device.
   * @param workspace_mr If provided, a separate memory resource to be used
   * for allocating temporary workspaces in RAFT calls.
   */
  static auto get_resources(int device_id = device_setter::get_current_device(),
                            std::shared_ptr<rmm::mr::device_memory_resource> workspace_mr = {
                              nullptr})
  {
    auto const& components = get_manager().get_device_components(device_id);
    return device_resources{components.get_stream(),
                            components.get_pool(),
                            workspace_mr ? workspace_mr : components.get_pool_memory_resource(),
                            components.get_workspace_allocation_limit()};
  }

  /**
   * @brief Set the total number of CUDA streams to be used per device
   *
   * If nullopt, the default stream per thread will be used
   * (essentially allowing as many streams as there are host threads).
   * Otherwise, all returned `device_resources` will draw their streams from this
   * limited pool.
   *
   * Limiting the total number of streams can be desirable for a number of
   * reasons, but it is most often used in consuming applications to
   * prevent a large number of host threads from flooding the device with
   * simultaneous requests that may exhaust device memory or other
   * resources.
   *
   * If called after the first call to
   * `raft::resource_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_streams_per_device(std::optional<std::size_t> num_streams)
  {
    get_manager().set_streams_per_device_(num_streams);
  }

  /**
   * @brief Set the total number and size of CUDA stream pools to be used per device
   *
   * Setting the number of stream pools to a non-zero value will provide a
   * pool of stream pools that can be shared among host threads. This can be
   * useful for the same reason it is useful to limit the total number of
   * primary streams assigned to `device_resoures` for each host thread.
   * Repeated calls to `get_device_resources` on a given host thread are
   * guaranteed to return `device_resources` with the same underlying stream
   * pool.
   *
   * If called after the first call to
   * `raft::resource_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_stream_pools_per_device(
    std::size_t num_pools, std::size_t num_streams = rmm::cuda_stream_pool::default_size)
  {
    get_manager().set_stream_pools_per_device_(num_pools, num_streams);
  }
  /**
   * @brief Set the maximum size of temporary RAFT workspaces
   *
   * Note that this limits only the size of temporary workspace
   * allocations. To cap the device memory generally available for all device
   * allocations made with RMM, use
   * `raft::device_manager::set_max_mem_pool_size`
   *
   * If called after the first call to
   * `raft::resource_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_workspace_allocation_limit(std::size_t memory_limit)
  {
    get_manager().set_workspace_allocation_limit_(memory_limit);
  }

  /**
   * @brief Set the maximum size of the device memory pool
   *
   * If set to 0, no memory pool will be used. If set to nullopt, the memory
   * pool is allowed to grow to the size of available device memory.
   *
   * Note that the pool will not actually be created until the first call
   * to `raft::device_manager::get_resources(device_id)`, after which it will become
   * the current RMM device memory resource for the indicated device. If the
   * current RMM device memory resource has already been set to some
   * non-default resource, no pool resource will be created and a warning will be emitted. It is
   * assumed that applications which have set a memory resource already wish to manage RMM
   * themselves.
   *
   * If called after the first call to
   * `raft::resource_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_max_mem_pool_size(std::optional<std::size_t> max_mem)
  {
    get_manager().set_max_mem_pool_size_(max_mem);
  }

  /**
   * @brief Set the initial size of the device memory pool
   *
   * If set to nullopt, the memory pool starts with half of the available
   * device memory.
   *
   * If called after the first call to
   * `raft::resource_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_init_mem_pool_size(std::optional<std::size_t> init_mem)
  {
    get_manager().set_init_mem_pool_size_(init_mem);
  }
  /**
   * @brief Request a device memory pool with specified parameters
   *
   * This convenience method essentially combines
   * `set_init_mem_pool_size` and `set_max_mem_pool_size`. It is provided
   * primarily to allow users who want a memory pool but do not want to choose
   * specific pool sizes to simply call
   * `raft::device_manager::set_memory_pool()` and enable a memory pool using
   * RMM defaults (initialize with half of available memory, allow to grow
   * to all available memory).
   *
   * If called after the first call to
   * `raft::resource_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_mem_pool(std::optional<std::size_t> init_mem = std::nullopt,
                           std::optional<std::size_t> max_mem  = std::nullopt)
  {
    set_init_mem_pool_size(init_mem);
    set_max_mem_pool_size(max_mem);
  }

  /**
   * @brief Ensure that all device work previously submitted by this thread
   * is complete before continuing host execution
   *
   * Note that this method *only* guarantees synchronization of work
   * submitted using the `device_resources` provided by
   * `raft::resource_manager::get_device_resources`. If `device_resources` are
   * created independent of the resource manager, work submitted using those
   * resources will not be synchronized.
   */
  static void synchronize_work_from_this_thread(int device_id = device_setter::get_current_device())
  {
    auto scoped_device = device_setter{device_id};
    auto res           = get_resources();
    res.sync_stream();
    if (res.is_stream_pool_initialized()) { res.sync_stream_pool(); }
  }

  /**
   * @brief Ensure that all device work previously submitted by any host thread
   * is complete before continuing host execution
   *
   * Note that this method *only* guarantees synchronization of work
   * submitted using the `device_resources` provided by
   * `raft::resource_manager::get_device_resources`. If `device_resources` are
   * created independent of the resource manager, work submitted using those
   * resources will not be synchronized.
   *
   * While similar in spirit to a `cudaDeviceSynchronize` call, this method
   * does not necessarily synchronize the entire device. Instead it synchronizes all streams for
   * this device managed by the resource manager. If the resource manager
   * was not configured to allocate streams from a limited pool, device
   * synchronization must be used. The advantage of this is that if some other
   * library is in use which manages its own streams, work submitted by that
   * library will not necessarily be synchronized. If synchronization of
   * *all* device work is required, `cudaDeviceSynchronize` should be
   * invoked directly.
   */
  static void synchronize_work_from_all_threads(int device_id = device_setter::get_current_device())
  {
    auto scoped_device     = device_setter{device_id};
    auto const& components = get_manager().get_device_components(device_id);
    if (components.stream_count() == 0) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());
    } else {
      components.synchronize();
    }
  }
};
}  // namespace raft
