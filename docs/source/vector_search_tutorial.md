# Vector Search in C++ Tutorial

RAFT has several important algorithms for performing vector search on the GPU and this tutorial walks through the primary vector search APIs from start to finish to provide a reference for quick setup and C++ API usage.

This tutorial assumes RAFT has been installed and/or added to your build so that you are able to compile and run RAFT code. If not done already, please follow the [build and install instructions](build.md) and consider taking a look at the [example c++ template project](https://github.com/rapidsai/raft/tree/HEAD/cpp/template) for ready-to-go examples that you can immediately build and start playing with. Also take a look at RAFT's library of [reproducible vector search benchmarks](raft_ann_benchmarks.md) to run benchmarks that compare RAFT against other state-of-the-art nearest neighbors algorithms at scale.

For more information about the various APIs demonstrated in this tutorial, along with comprehensive usage examples of all the APIs offered by RAFT, please refer to the [RAFT's C++ API Documentation](https://docs.rapids.ai/api/raft/nightly/cpp_api/). 

## Step 1: Starting off with RAFT

### CUDA Development? 

If you are reading this tuturial then you probably know about CUDA and its relationship to general-purpose GPU computing (GPGPU). You probably also know about Nvidia GPUs but might not necessarily be familiar with the programming model nor GPU computing. The good news is that extensive knowledge of CUDA and GPUs are not needed in order to get started with or build applications with RAFT. RAFT hides away most of the complexities behind simple single-threaded stateless functions that are inherently asynchronous, meaning the result of a computation isn't necessarily read to be used when the function executes and control is given back to the user. The functions are, however, allowed to be chained together in a sequence of calls that don't need to wait for subsequent computations to complete in order to continue execution. In fact, the only time you need to wait for the computation to complete is when you are ready to use the result.

A common structure you will encounter when using RAFT is a `raft::device_resources` object. This object is a container for important resources for a single GPU that might be needed during computation. If communicating with multiple GPUs, multiple `device_resources` might be needed, one for each GPU. `device_resources` contains several methods for managing its state but most commonly, you'll call the `sync_stream()` to guarantee all recently submitted computation has completed (as mentioned above.)

A simple example of using `raft::device_resources` in RAFT:

```c++
#include <raft/core/device_resources.hpp>

raft::device_resources res;
// Call a bunch of RAFT functions in sequence...
res.sync_stream()
```

### Host vs Device Memory

We differentiate between two different types of memory. `host` memory is your traditional RAM memory that is primarily accessible by applications on the CPU. `device` memory, on the other hand, is what we call the special memory on the GPU, which is not accessible from the CPU. In order to access host memory from the GPU, it needs to be explicitly copied to the GPU and in order to access device memory by the CPU, it needs to be explicitly copied there. We have several mechanisms available for allocating and managing the lifetime of device memory on the stack so that we don't need to explicitly allocate and free pointers on the heap. For example, instead of a `std::vector` for host memory, we can use `rmm::device_uvector` on the device. The following function will copy an array from host memory to device memory:

```c++
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

raft::device_resources res;

std::vector<int> my_host_vector = {0, 1, 2, 3, 4};
rmm::device_uvector<int> my_device_vector(my_host_vector.size(), res.get_stream());

raft::copy(my_device_vector.data(), my_host_vector.data(), my_host_vector.size(), res.get_stream());
```

Since a stream is involved in the copy operation above, RAFT functions can be invoked immediately so long as the same `device_resources` instances is used (or, more specifically, the same main stream from the `devices_resources`.) As you might notice in the example above, `res.get_stream()` can be used to extract the main stream from a `device_resources` instance.

### Multi-dimensional data representation

`rmm::device_uvector` is a great mechanism for allocating and managing a chunk of device memory. While it's possible to use a single array to represent objects in higher dimensions like matrices, it lacks the means to pass that information along. For example, in addition to knowing that we have a 2d structure, we would need to know the number of rows, the number of columns, and even whether we read the columns or rows first (referred to as column- or row-major respectively).

For this reason, RAFT relies on the `mdspan` standard, which was composed specifically for this purpose. To be even more, `mdspan` itself doesn't actually allocate or own any data on host or device because it's just a view over an existing memory on host device. The `mdspan` simply gives us a way to represent multi-dimensional data so we can pass along the needed metadata to our APIs. Even more powerful is that we can design functions that only accept a matrix of `float` in device memory that is laid out in row-major format. 

The memory-owning counterpart to the `mdspan` is the `mdarray` and the `mdarray` can allocate memory on device or host and carry along with it the metadata about its shape and layout. An `mdspan` can be produced from an `mdarray` for invoking RAFT APIs with `mdarray.view()`. They also follow similar paradigms to the STL, where we represent an immutable `mdspan` of `int` using `mdspan<const int>` instead of `const mdspan<int>` to ensure it's the type carried along by the `mdspan` that's not allowed to change. 

Many RAFT functions require `mdspan<const T>` to represent immutable input data and there's no implicit conversion between `mdspan<T>` and `mdspan<const T>` we use `raft::make_const_mdspan()` to alleviate the pain of constructing a new `mdspan` to invoke these functions.

The following example demonstrates how to create `mdarray` matrices in both device and host memory, copy one to the other, and create mdspans out of them:

```c++
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/copy.hpp>

raft::device_resources res;

int n_rows = 10;
int n_cols = 10;

auto device_matrix = raft::make_device_matrix<float>(res, n_rows, n_cols);
auto host_matrix = raft::make_host_matrix<float>(res, n_rows, n_cols);

// Set the diagonal to 1
for(int i = 0; i < n_rows; i++) {
    host_matrix(i, i) = 1;
}

raft::copy(res, device_matrix.view(), host_matrix.view());
```

## Step 2: Generate some data

Let's build upon the fundamentals from the prior section and actually invoke some of RAFT's computational APIs on the device. A good starting point is data generation.

```c++
#include <raft/core/device_mdarray.hpp>
#include <raft/random/make_blobs.cuh>

raft::device_resources res;

int n_rows = 10000;
int n_cols = 10000;

auto dataset = raft::make_device_matrix<float, int>(res, n_rows, n_cols);
auto labels = raft::make_device_vector<float, int>(res, n_rows);

raft::make_blobs(res, dataset.view(), labels.view());
```

That's it. We've now generated a random 10kx10k matrix with points that cleanly separate into Gaussian clusters, along with a vector of cluster labels for each of the data points. Notice the `cuh` extension in the header file include for `make_blobs`. This signifies to us that this file contains CUDA device functions like kernel code so the CUDA compiler, `nvcc` is needed in order to compile any code that uses it. Generally, any source files that include headers with a `cuh` extension use the `.cu` extension instead of `.cpp`. The rule here is that `cpp` source files contain code which can be compiled with a C++ compiler like `g++` while `cu` files require the CUDA compiler.

Since the `make_blobs` code generates the random dataset on the GPU device, we didn't need to do any host to device copies in this one. `make_blobs` is also asynchronous, so if we don't need to copy and use the data in host memory right away, we can continue calling RAFT functions with the `device_resources` instance and the data transformations will all be scheduled on the same stream.

## Step 3: Using brute-force indexes

### Build brute-force index

Consider the `(10k, 10k)` shaped random matrix we generated in the previous step. We want to be able to find the k-nearest neighbors for all points of the matrix, or what we refer to as the all-neighbors graph, which means finding the neighbors of all data points within the same matrix.
```c++
#include <raft/neighbors/brute_force.cuh>

raft::device_resources res;

// set number of neighbors to search for
int const k = 64;

auto bfknn_index = raft::neighbors::brute_force::build(res,
                                                       raft::make_const_mdspan(dataset.view()));
```

### Query brute-force index

```c++

// using matrix `dataset` from previous example
auto search = raft::make_const_mdspan(dataset.view());

// Indices and Distances are of dimensions (n, k)
// where n is number of rows in the search matrix
auto reference_indices = raft::make_device_matrix<int, int>(search.extent(0), k); // stores index of neighbors
auto reference_distances = raft::make_device_matrix<float, int>(search.extent(0), k); // stores distance to neighbors

raft::neighbors::brute_force::search(res,
                                     bfknn_index,
                                     search,
                                     raft::make_const_mdspan(indices.view()),
                                     raft::make_const_mdspan(distances.view()));
```

We have established several things here by building a flat index. Now we know the exact 64 neighbors of all points in the matrix, and this algorithm can be generally useful in several ways:
1. Creating a baseline to compare against when building an approximate nearest neighbors index.
2. Directly using the brute-force algorithm when accuracy is more important than speed of computation. Don't worry, our implementation is still the best in-class and will provide not only significant speedups over other brute force methods, but also be quick relatively when the matrices are small!


## Step 4: Using the ANN indexes

### Build a CAGRA index

Next we'll train an ANN index. We'll use our graph-based CAGRA algorithm for this example but the other index types use a very similar pattern.

```c++
#include <raft/neighbors/cagra.cuh>

raft::device_resources res;

// use default index parameters
cagra::index_params index_params;

auto index = cagra::build<float, uint32_t>(res, index_params, dataset);
```

### Query the CAGRA index

Now that we've trained a CAGRA index, we can query it by first allocating our output `mdarray` objects and passing the trained index model into the search function. 

```c++
// create output arrays
auto indices = raft::make_device_matrix<uint32_t>(res, n_rows, k);
auto distances = raft::make_device_matrix<float>(res, n_rows, k);

// use default search parameters
cagra::search_params search_params;

// search K nearest neighbors
cagra::search<float, uint32_t>(
res, search_params, index, search, indices.view(), distances.view());
```

## Step 7: Evaluate neighborhood quality

In step 3 we built a flat index and queried for exact neighbors while in step 4 we build an ANN index and queried for approximate neighbors. How do you quickly figure out the quality of our approximate neighbors and whether it's in an acceptable range based on your needs? Just compute the `neighborhood_recall` which gives a single value in the range [0, 1]. Closer the value to 1, higher the quality of the approximation.

```c++
#include <raft/stats/neighborhood_recall.cuh>

raft::device_resources res;

// Assuming matrices as type raft::device_matrix_view and variables as
// indices : approximate neighbor indices
// reference_indices : exact neighbor indices
// distances : approximate neighbor distances
// reference_distances : exact neighbor distances

// We want our `neighborhood_recall` value in host memory
float const recall_scalar = 0.0;
auto recall_value = raft::make_host_scalar(recall_scalar);

raft::stats::neighborhood_recall(res,
                                 raft::make_const_mdspan(indices.view()),
                                 raft::make_const_mdspan(reference_indices.view()),
                                 recall_value.view(),
                                 raft::make_const_mdspan(distances),
                                 raft::make_const_mdspan(reference_distances));

res.sync_stream();
```

Notice we can run invoke the functions for index build and search for both algorithms, one right after the other, because we don't need to access any outputs from the algorithms in host memory. We will need to synchronize the stream on the `raft::device_resources` instance before we can read the result of the `neighborhood_recall` computation, though. 

Similar to a Numpy array, when we use a `host_scalar`, we are really using a multi-dimensional structure that contains only a single dimension, and further a single element. We can use element indexing to access the resulting element directly.
```c++
std::cout << recall_value(0) << std::endl;
```

While it may seem like unnecessary additional work to wrap the result in a `host_scalar` mdspan, this API choice is made intentionally to support the possibility of also receiving the result as a `device_scalar` so that it can be used directly on the device for follow-on computations without having to incur the synchronization or transfer cost of bringing the result to host. This pattern becomes even more important when the result is being computed in a loop, such as an iterative solver, and the cost of synchronization and device-to-host (d2h) transfer becomes very expensive. 

## Advanced features

The following sections present some advanced features that we have found can be useful for squeezing more utilization out of GPU hardware. As you've seen in this tutorial, RAFT provides several very useful tools and building blocks for developing accelerated applications beyond vector search capabilities.

### Stream pools

Within each CPU thread, CUDA uses `streams` to submit asynchronous work. You can think of a stream as a queue. Each stream can submit work to the GPU independently of other streams but work submitted within each stream is queued and executed in the order in which it was submitted. Similar to how we can use thread pools to bound the parallelism of CPU threads, we can use CUDA stream pools to bound the amount of concurrent asynchronous work that can be scheduled on a GPU. Each instance of `device_resources` has a main stream, but can also create a stream pool. For a single CPU thread, multiple different instances of `device_resources` can be created with different main streams and used to invoke a series of RAFT functions concurrently on the same or different GPU devices, so long as the target devices have available resources to perform the work. Once a device is saturated, queued work on streams will be scheduled and wait for a chance to do more work. During this time the streams are waiting, the CPU thread will still continue its own execution asynchronously unless `sync_stream_pool()` is called, causing the thread to block and wait for the thread pools to complete.

Also, beware that before splitting GPU work onto multiple different concurrent streams, it can often be important to wait for the main stream in the `device_resources`. This can be done with `wait_stream_pool_on_stream()`.

To summarize, if wanting to execute multiple different streams in parallel, we would often use a stream pool like this:
```c++
#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream.hpp>

int n_streams = 5;

rmm::cuda_stream stream;
std::shared_ptr<rmm::cuda_stream_pool> stream_pool(5)
raft::device_resources res(stream.view(), stream_pool);

// Submit some work on the main stream...

res.wait_stream_pool_on_stream()
for(int i = 0; i < n_streams; ++i) {
    rmm::cuda_stream_view stream_from_pool = res.get_next_usable_stream();
    raft::device_resources pool_res(stream_from_pool);
    // Submit some work with pool_res...
}

res.sync_stream_pool();
```

### Device resources manager

In multi-threaded applications, it is often useful to create a set of
`raft::device_resources` objects on startup to avoid the overhead of
re-initializing underlying resources every time a `raft::device_resources` object
is needed. To help simplify this common initialization logic, RAFT
provides a `raft::device_resources_manager` to handle this for downstream
applications. On startup, the application can specify certain limits on the
total resource consumption of the `raft::device_resources` objects that will be
generated:
```c++
#include <raft/core/device_resources_manager.hpp>

void initialize_application() {
  // Set the total number of CUDA streams to use on each GPU across all CPU
  // threads. If this method is not called, the default stream per thread
  // will be used.
  raft::device_resources_manager::set_streams_per_device(16);

  // Create a memory pool with given max size in bytes. Passing std::nullopt will allow
  // the pool to grow to the available memory of the device.
  raft::device_resources_manager::set_max_mem_pool_size(std::nullopt);

  // Set the initial size of the memory pool in bytes.
  raft::device_resources_manager::set_init_mem_pool_size(16000000);

  // If neither of the above methods are called, no memory pool will be used
}
```
While this example shows some commonly used settings,
`raft::device_resources_manager` provides support for several other
resource options and constraints, including options to initialize entire
stream pools that can be used by an individual `raft::device_resources` object. After
this initialization method is called, the following function could be called
from any CPU thread:
```c++
void foo() {
  raft::device_resources const& res = raft::device_resources_manager::get_device_resources();
  // Submit some work with res
  res.sync_stream();
}
```

If any `raft::device_resources_manager` setters are called _after_ the first
call to `raft::device_resources_manager::get_device_resources()`, these new
settings are ignored, and a warning will be logged. If a thread calls
`raft::device_resources_manager::get_device_resources()` multiple times, it is
guaranteed to access the same underlying `raft::device_resources` object every
time. This can be useful for chaining work in different calls on the same
thread without keeping a persistent reference to the resources object.

### Device memory resources

The RAPIDS software ecosystem makes heavy use of the [RAPIDS Memory Manager](https://github.com/rapidsai/rmm) (RMM) to enable zero-copy sharing of device memory across various GPU-enabled libraries such as PyTorch, Jax, Tensorflow, and FAISS. A really powerful feature of RMM is the ability to set a memory resource, such as a pooled memory resource that allocates a block of memory up front to speed up subsequent smaller allocations, and have all the libraries in the GPU ecosystem recognize and use that same memory resource for all of their memory allocations.

As an example, the following code snippet creates a `pool_memory_resource` and sets it as the default memory resource, which means all other libraries that use RMM will now allocate their device memory from this same pool:
```c++
#include <rmm/mr/device/pool_memory_resource.hpp>

rmm::mr::cuda_memory_resource cuda_mr;
// Construct a resource that uses a coalescing best-fit pool allocator
rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_mr{&cuda_mr};
rmm::mr::set_current_device_resource(&pool_mr); // Updates the current device resource pointer to `pool_mr`
```

The `raft::device_resources` object will now also use the `rmm::current_device_resource`.  This isn't limited to C++, however. Often a user will be interacting with PyTorch, RAPIDS, or Tensorflow through Python and so they can set and use RMM's `current_device_resource` [right in Python](https://github.com/rapidsai/rmm#using-rmm-in-python-code).

### Workspace memory resource

As mentioned above, `raft::device_resources` will use `rmm::current_device_resource` by default for all memory allocations. However, there are times when a particular algorithm might benefit from using a different memory resource such as a `managed_memory_resource`, which creates a unified memory space between device and host memory, paging memory in and out of device as needed. Most of RAFT's algorithms allocate temporary memory as needed to perform their computations and we can control the memory resource used for these temporary allocations through the `workspace_resource` in the `raft::device_resources` instance. 

For some applications, the `managed_memory_resource`, can enable a memory space that is larger than the GPU, thus allowing a natural spilling to host memory when needed. This isn't always the best way to use managed memory, though, as it can quickly lead to thrashing and severely impact performance. Still, when it can be used, it provides a very powerful tool that can also avoid out of memory errors when enough host memory is available. 

The following creates a managed memory allocator and set it as the `workspace_resource` of the `raft::device_resources` instance:
```c++
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

std::shared_ptr<rmm::mr::managed_memory_resource> managed_resource;
raft::device_resource res(managed_resource);
```

The `workspace_resource` uses an `rmm::mr::limiting_resource_adaptor`, which limits the total amount of allocation possible. This allows RAFT algorithms to work within the confines of the memory constraints imposed by the user so that things like batch sizes can be automatically set to reasonable values without exceeding the allotted memory. By default, this limit restricts the memory allocation space for temporary workspace buffers to the memory available on the device. 

The below example specifies the total number of bytes that RAFT can use for temporary workspace allocations to 3GB:
```c++
#include <raft/core/device_resources.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>

#include <optional>

std::shared_ptr<rmm::mr::managed_memory_resource> managed_resource;
raft::device_resource res(managed_resource, std::make_optional<std::size_t>(3 * 1024^3));
```