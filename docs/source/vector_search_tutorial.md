# Vector Search in C++ Tutorial

RAFT has several important algorithms for performing vector search on the GPU and this tutorial walks through the primary vector search APIs from start to finish to provide a reference for quick setup and C++ API usage.

This tutorial assumes RAFT has been installed and/or added to your build so that you are able to compile and run RAFT code. If not done already, please follow the [build and install instructions](build.md) and consider taking a look at the [example c++ template project](https://github.com/rapidsai/raft/tree/HEAD/cpp/template) for a ready-to-go example that you can immediately build and start playing with. Also take a look at RAFT's library of [reproducible vector search benchmarks](raft_ann_benchmarks.md) to run benchmarks that compare RAFT against other state-of-the-art nearest neighbors algorithms at scale.


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

In multi-threaded applications, it is often useful to create a set of
`raft::device_resources` objects on startup to avoid the overhead of
re-initializing underlying resources every time a `raft::device_resources` object
is needed. To help simplify this common initialization logic, RAFT
provides a `raft::device_resources_manager` to handle this for downstream
applications. On startup, the application can specify certain limits on the
total resource consumption of the `raft::device_resources` objects that will be
generated:
```
#include <raft/core/device_resources_manager>

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
```
#include <raft/core/device_resources_manager>
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
#include <raft/core/device_resources.hpp>

raft::device_resources res;

int n_rows = 10;
int n_cols = 10;

auto device_matrix = raft::make_device_matrix<float>(res, n_rows, n_cols);
auto host_matrix = raft::make_host_matrix<float>(res, n_rows, n_cols);

// Set the diagonal to 1
for(int i = 0; i < n_rows; i++) {
    host_matrix(i, i) = 1;
}

raft::copy(device_matrix.data_handle(), host_matrix.data_handle(), host_matrix.size(), res.get_stream());
```

## Step 2: Generate some data

Let's build upon the fundamentals from the prior section and actually invoke some of RAFT's computational APIs on the device. A good starting point is data generation.

```c++
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
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



## Step 3: Calculate exact nearest neighbors
Consider the 10kx10k random matrix we generated in the previous step. We want to be able to find the k-nearest neighbors for all points of the matrix, or what we refer to as the all-neighbors graph which means finding the neighbors of a data point within the same matrix.
```c++
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/brute_force.cuh>

raft::device_resources res;

// set number of neighbors to search for
int const k = 64;

// using matrix `dataset` from previous example
std::vector<raft::device_matrix_view<const float, int>> index(raft::make_const_mdspan(dataset.view()));
auto search = raft::make_const_mdspan(dataset.view());

// Indices and Distances are of dimensions (n, k)
// where n is number of rows in the search matrix
auto indices = raft::make_device_matrix<int, int>(search.extent(0), k); // stores index of neighbors
auto distances = raft::make_device_matrix<float, int>(search.extent(0), k); // stores distance to neighbors

// Compute exact-neighbors using Euclidean distance
raft::neighbors::brute_force::knn(index,
                                  search,
                                  indices,
                                  distances,
                                  raft::distance::DistanceType::L2Unexpanded);
```

We have established several things here by building a flat index. Now we know the exact 64 neighbors of all points in the matrix, and this algorithm can be generally useful in several ways:
1. Creating a baseline to compare against when building an Approximate Nearest Neighbors index.
2. Directly using the brute force algorithm when accuracy is more important than speed of computation. Don't worry, our implementation is still the best in-class and will provide not only significant speedups over other brute force methods, but also be quick relatively when the matrices are small!


## Step 4: Train an ANN index

Now comes the fun part of training



## Step 5: Add vectors to index

## Step 6: Query the index

## Step 7: Additional features
### Comparing exact and approximate neighbor quality
In step 3 we built a flat index and queried for exact neighbors while in step 4 we build an ANN index and queried for approximate neighbors. How do you quickly figure out the quality of our approximate neighbors and whether it's in an acceptable range based on your needs? Just compute the `neighborhood_recall` which gives a single value in the range [0, 1]. Closer the value to 1, higher the quality of the approximation.

```c++
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
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
                                 indices,
                                 reference_indices,
                                 recall_value.view(),
                                 distances,
                                 reference_distances);
```
