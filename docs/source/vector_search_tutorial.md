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

All that's been mentioned so far applies to a single CPU thread. RAFT provides a `raft::device_resources_manager` that can manage streams and stream pools for multiple devices across multiple threads.

TODO: Examples and better explanation about `device_resources_manager`

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

The following example demonstrates how to create `mdarray` matrices in device and host memory, and create mdspans out of them:

```c++

```


## Step 2: Generate some data

Let's build upon the fundamental from the prior section andi fundamentals of RAFT API call patterns, let's 

## Step 3: Train an ANN index

## Step 4: Add vectors to index

## Step 5: Query the index

## Step 6: Additional features

