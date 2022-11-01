# Quick Start

This guide is meant to provide a quick-start tutorial for interacting with RAFT's C++ APIs.

## RAPIDS Memory Manager (RMM)

RAFT relies heavily on the [RMM](https://github.com/rapidsai/rmm) library which eases the burden of configuring different allocation strategies globally across the libraries that use it.

## Multi-dimensional Spans and Arrays

The APIs in RAFT currently accept raw pointers to device memory and we are in the process of simplifying the APIs with the [mdspan](https://arxiv.org/abs/2010.06474) multi-dimensional array view for representing data in higher dimensions similar to the `ndarray` in the Numpy Python library. RAFT also contains the corresponding owning `mdarray` structure, which simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory.

The `mdarray` forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:

```c++
#include <raft/core/device_mdarray.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_device_scalar<float>(handle, 1.0);
auto vector = raft::make_device_vector<float>(handle, n_cols);
auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);
```

The `mdspan` is a lightweight non-owning view that can wrap around any pointer, maintaining shape, layout, and indexing information for accessing elements. 


We can construct `mdspan` instances directly from the above `mdarray` instances:

```c++
// Scalar mdspan on device
auto scalar_view = scalar.view();

// Vector mdspan on device
auto vector_view = vector.view();

// Matrix mdspan on device
auto matrix_view = matrix.view();
```
Since the `mdspan` is just a lightweight wrapper, we can also construct it from the underlying data handles in the `mdarray` instances above. We use the extent to get information about the `mdarray` or `mdspan`'s shape.

```c++
#include <raft/core/device_mdspan.hpp>

auto scalar_view = raft::make_device_scalar_view(scalar.data_handle());
auto vector_view = raft::make_device_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_device_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```

Of course, RAFT's `mdspan`/`mdarray` APIs aren't just limited to the `device`. You can also create `host` variants:

```c++
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_host_scalar<float>(handle, 1.0);
auto vector = raft::make_host_vector<float>(handle, n_cols);
auto matrix = raft::make_host_matrix<float>(handle, n_rows, n_cols);

auto scalar_view = raft::make_host_scalar_view(scalar.data_handle());
auto vector_view = raft::make_host_vector_view(vector.data_handle(), vector.extent(0));
auto matrix_view = raft::make_host_matrix_view(matrix.data_handle(), matrix.extent(0), matrix.extent(1));
```

And `managed` variants:

```c++
#include <raft/core/device_mdspan.hpp>

int n_rows = 10;
int n_cols = 10;

auto matrix = raft::make_managed_mdspan(managed_ptr, raft::make_matrix_extents(n_rows, n_cols));
```


## C++ Example

Most of the primitives in RAFT accept a `raft::handle_t` object for the management of resources which are expensive to create, such CUDA streams, stream pools, and handles to other CUDA libraries like `cublas` and `cusolver`.

The example below demonstrates creating a RAFT handle and using it with `device_matrix` and `device_vector` to allocate memory, generating random clusters, and computing
pairwise Euclidean distances:

```c++
#include <raft/core/handle.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/distance/distance.cuh>

raft::handle_t handle;

int n_samples = 5000;
int n_features = 50;

auto input = raft::make_device_matrix<float>(handle, n_samples, n_features);
auto labels = raft::make_device_vector<int>(handle, n_samples);
auto output = raft::make_device_matrix<float>(handle, n_samples, n_samples);

raft::random::make_blobs(handle, input.view(), labels.view());

auto metric = raft::distance::DistanceType::L2SqrtExpanded;
raft::distance::pairwise_distance(handle, input.view(), input.view(), output.view(), metric);
```

## Python Example

The `pylibraft` package contains a Python API for RAFT algorithms and primitives. `pylibraft` integrates nicely into other libraries by being very lightweight with minimal dependencies and accepting any object that supports the `__cuda_array_interface__`, such as [CuPy's ndarray](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#rmm). The package is currently limited to pairwise distances and RMAT graph generation, but we will continue adding more in future releases.

The example below demonstrates computing the pairwise Euclidean distances between CuPy arrays. `pylibraft` is a low-level API that prioritizes efficiency and simplicity over being pythonic, which is shown here by pre-allocating the output memory before invoking the `pairwise_distance` function. Note that CuPy is not a required dependency for `pylibraft`.

```python
import cupy as cp

from pylibraft.distance import pairwise_distance

n_samples = 5000
n_features = 50

in1 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
in2 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
output = cp.empty((n_samples, n_samples), dtype=cp.float32)

pairwise_distance(in1, in2, output, metric="euclidean")
```
