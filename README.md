# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp;RAFT: Reusable Accelerated Functions and Tools for Vector Search and More</div>

![RAFT tech stack](img/raft-tech-stack-vss.png)


## Contents
<hr>

1. [Useful Resources](#useful-resources)
2. [What is RAFT?](#what-is-raft)
2. [Use cases](#use-cases)
3. [Is RAFT right for me?](#is-raft-right-for-me)
4. [Getting Started](#getting-started)
5. [Installing RAFT](#installing)
6. [Codebase structure and contents](#folder-structure-and-contents)
7. [Contributing](#contributing)
8. [References](#references)

<hr>

## Useful Resources

- [RAFT Reference Documentation](https://docs.rapids.ai/api/raft/stable/): API Documentation.
- [RAFT Getting Started](./docs/source/quick_start.md): Getting started with RAFT.
- [Build and Install RAFT](./docs/source/build.md): Instructions for installing and building RAFT.
- [Example Notebooks](./notebooks): Example jupyer notebooks
- [RAPIDS Community](https://rapids.ai/community.html): Get help, contribute, and collaborate.
- [GitHub repository](https://github.com/rapidsai/raft): Download the RAFT source code.
- [Issue tracker](https://github.com/rapidsai/raft/issues): Report issues or request features.



## What is RAFT?

RAFT contains fundamental widely-used algorithms and primitives for machine learning and information retrieval. The algorithms are CUDA-accelerated and form building blocks for more easily writing high performance applications.

By taking a primitives-based approach to algorithm development, RAFT
- accelerates algorithm construction time
- reduces the maintenance burden by maximizing reuse across projects, and
- centralizes core reusable computations, allowing future optimizations to benefit all algorithms that use them.

While not exhaustive, the following general categories help summarize the accelerated functions in RAFT:
#####
| Category              | Accelerated Functions in RAFT                                                                                                     |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Nearest Neighbors** | vector search, neighborhood graph construction, epsilon neighborhoods, pairwise distances                                         |
| **Basic Clustering**  | spectral clustering, hierarchical clustering, k-means                                                                             |
| **Solvers**           | combinatorial optimization, iterative solvers                                                                                     |
| **Data Formats**      | sparse & dense, conversions, data generation                                                                                      |
| **Dense Operations**  | linear algebra, matrix and vector operations, reductions, slicing, norms, factorization, least squares, svd & eigenvalue problems |
| **Sparse Operations** | linear algebra, eigenvalue problems, slicing, norms, reductions, factorization, symmetrization, components & labeling             |
| **Statistics**        | sampling, moments and summary statistics, metrics, model evaluation                                                               |
| **Tools & Utilities** | common tools and utilities for developing CUDA applications, multi-node multi-gpu infrastructure                                  |


RAFT is a C++ header-only template library with an optional shared library that
1) can speed up compile times for common template types, and
2) provides host-accessible "runtime" APIs, which don't require a CUDA compiler to use

In addition being a C++ library, RAFT also provides 2 Python libraries:
- `pylibraft` - lightweight Python wrappers around RAFT's host-accessible "runtime" APIs.
- `raft-dask` - multi-node multi-GPU communicator infrastructure for building distributed algorithms on the GPU with Dask.

![RAFT is a C++ header-only template library with optional shared library and lightweight Python wrappers](img/arch.png)

## Use cases

### Vector Similarity Search

RAFT contains state-of-the-art implementations of approximate nearest neighbors search (ANNS) algorithms on the GPU, such as:

* [Brute force](https://docs.rapids.ai/api/raft/nightly/pylibraft_api/neighbors/#brute-force). Performs a brute force nearest neighbors search without an index.
* [IVF-Flat](https://docs.rapids.ai/api/raft/nightly/pylibraft_api/neighbors/#ivf-flat) and [IVF-PQ](https://docs.rapids.ai/api/raft/nightly/pylibraft_api/neighbors/#ivf-pq). Use an inverted file index structure to map contents to their locations. IVF-PQ additionally uses product quantization to reduce the memory usage of vectors. These methods were originally popularized by the [FAISS](https://github.com/facebookresearch/faiss) library.
* [CAGRA](https://docs.rapids.ai/api/raft/nightly/pylibraft_api/neighbors/#cagra) (Cuda Anns GRAph-based). Uses a fast ANNS graph construction and search implementation optimized for the GPU. CAGRA outperforms state-of-the art CPU methods (i.e. HNSW) for large batch queries, single queries, and graph construction time. 

Projects that use the RAFT ANNS algorithms for accelerating vector search include: [Milvus](https://milvus.io/), [Redis](https://redis.io/), and [Faiss](https://github.com/facebookresearch/faiss). 

Please see the example [Jupyter notebook](https://github.com/rapidsai/raft/blob/HEAD/notebooks/VectorSearch_QuestionRetrieval.ipynb) to get started RAFT for vector search in Python.

### Information Retrieval

RAFT contains a catalog of reusable primitives for composing algorithms that require fast neighborhood computations, such as

1. Computing distances between vectors and computing kernel gramm matrices
2. Performing ball radius queries for constructing epsilon neighborhoods
3. Clustering points to partition a space for smaller and faster searches
4. Constructing neighborhood "connectivities" graphs from dense vectors

### Machine Learning

RAFT's primitives are used in several RAPIDS libraries, including [cuML](https://github.com/rapidsai/cuml), [cuGraph](https://github.com/rapidsai/cugraph), and [cuOpt](https://github.com/rapidsai/cuopt) to build many end-to-end machine learning algorithms that span a large spectrum of different applications, including 
- data generation 
- model evaluation
- classification and regression
- clustering
- manifold learning
- dimensionality reduction.

RAFT is also used by the popular collaborative filtering library [implicit](https://github.com/benfred/implicit) for recommender systems.

## Is RAFT right for me?

RAFT contains low-level primitives for accelerating applications and workflows. Data source providers and application developers may find specific tools -- like ANN algorithms -- very useful. RAFT is not intended to be used directly by data scientists for discovery and experimentation. For data science tools, please see the [RAPIDS website](https://rapids.ai/).

## Getting started

### RAPIDS Memory Manager (RMM)

RAFT relies heavily on RMM which eases the burden of configuring different allocation strategies globally across the libraries that use it.

### Multi-dimensional Arrays

The APIs in RAFT accept the [mdspan](https://arxiv.org/abs/2010.06474) multi-dimensional array view for representing data in higher dimensions similar to the `ndarray` in the Numpy Python library. RAFT also contains the corresponding owning `mdarray` structure, which simplifies the allocation and management of multi-dimensional data in both host and device (GPU) memory.

The `mdarray` forms a convenience layer over RMM and can be constructed in RAFT using a number of different helper functions:

```c++
#include <raft/core/device_mdarray.hpp>

int n_rows = 10;
int n_cols = 10;

auto scalar = raft::make_device_scalar<float>(handle, 1.0);
auto vector = raft::make_device_vector<float>(handle, n_cols);
auto matrix = raft::make_device_matrix<float>(handle, n_rows, n_cols);
```

### C++ Example

Most of the primitives in RAFT accept a `raft::device_resources` object for the management of resources which are expensive to create, such CUDA streams, stream pools, and handles to other CUDA libraries like `cublas` and `cusolver`.

The example below demonstrates creating a RAFT handle and using it with `device_matrix` and `device_vector` to allocate memory, generating random clusters, and computing
pairwise Euclidean distances:
```c++
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/distance/distance.cuh>

raft::device_resources handle;

int n_samples = 5000;
int n_features = 50;

auto input = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
auto labels = raft::make_device_vector<int, int>(handle, n_samples);
auto output = raft::make_device_matrix<float, int>(handle, n_samples, n_samples);

raft::random::make_blobs(handle, input.view(), labels.view());

auto metric = raft::distance::DistanceType::L2SqrtExpanded;
raft::distance::pairwise_distance(handle, input.view(), input.view(), output.view(), metric);
```

It's also possible to create `raft::device_mdspan` views to invoke the same API with raw pointers and shape information:

```c++
#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/distance/distance.cuh>

raft::device_resources handle;

int n_samples = 5000;
int n_features = 50;

float *input;
int *labels;
float *output;

...
// Allocate input, labels, and output pointers
...

auto input_view = raft::make_device_matrix_view(input, n_samples, n_features);
auto labels_view = raft::make_device_vector_view(labels, n_samples);
auto output_view = raft::make_device_matrix_view(output, n_samples, n_samples);

raft::random::make_blobs(handle, input_view, labels_view);

auto metric = raft::distance::DistanceType::L2SqrtExpanded;
raft::distance::pairwise_distance(handle, input_view, input_view, output_view, metric);
```


### Python Example

The `pylibraft` package contains a Python API for RAFT algorithms and primitives. `pylibraft` integrates nicely into other libraries by being very lightweight with minimal dependencies and accepting any object that supports the `__cuda_array_interface__`, such as [CuPy's ndarray](https://docs.cupy.dev/en/stable/user_guide/interoperability.html#rmm). The number of RAFT algorithms exposed in this package is continuing to grow from release to release.

The example below demonstrates computing the pairwise Euclidean distances between CuPy arrays. Note that CuPy is not a required dependency for `pylibraft`.

```python
import cupy as cp

from pylibraft.distance import pairwise_distance

n_samples = 5000
n_features = 50

in1 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
in2 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)

output = pairwise_distance(in1, in2, metric="euclidean")
```

The `output` array in the above example is of type `raft.common.device_ndarray`, which supports [__cuda_array_interface__](https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html#cuda-array-interface-version-2) making it interoperable with other libraries like CuPy, Numba, PyTorch and RAPIDS cuDF that also support it. CuPy supports DLPack, which also enables zero-copy conversion from `raft.common.device_ndarray` to JAX and Tensorflow.

Below is an example of converting the output `pylibraft.device_ndarray` to a CuPy array:
```python
cupy_array = cp.asarray(output)
```

And converting to a PyTorch tensor:
```python
import torch

torch_tensor = torch.as_tensor(output, device='cuda')
```

Or converting to a RAPIDS cuDF dataframe:
```python
cudf_dataframe = cudf.DataFrame(output)
```

When the corresponding library has been installed and available in your environment, this conversion can also be done automatically by all RAFT compute APIs by setting a global configuration option:
```python
import pylibraft.config
pylibraft.config.set_output_as("cupy")  # All compute APIs will return cupy arrays
pylibraft.config.set_output_as("torch") # All compute APIs will return torch tensors
```

You can also specify a `callable` that accepts a `pylibraft.common.device_ndarray` and performs a custom conversion. The following example converts all output to `numpy` arrays:
```python
pylibraft.config.set_output_as(lambda device_ndarray: return device_ndarray.copy_to_host())
```

`pylibraft` also supports writing to a pre-allocated output array so any `__cuda_array_interface__` supported array can be written to in-place:

```python
import cupy as cp

from pylibraft.distance import pairwise_distance

n_samples = 5000
n_features = 50

in1 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
in2 = cp.random.random_sample((n_samples, n_features), dtype=cp.float32)
output = cp.empty((n_samples, n_samples), dtype=cp.float32)

pairwise_distance(in1, in2, out=output, metric="euclidean")
```


## Installing

RAFT's C++ and Python libraries can both be installed through Conda and the Python libraries through Pip. 


### Installing C++ and Python through Conda

The easiest way to install RAFT is through conda and several packages are provided.
- `libraft-headers` C++ headers
- `libraft` (optional) C++ shared library containing pre-compiled template instantiations and runtime API.
- `pylibraft` (optional) Python library
- `raft-dask` (optional) Python library for deployment of multi-node multi-GPU algorithms that use the RAFT `raft::comms` abstraction layer in Dask clusters.
- `raft-ann-bench` (optional) Benchmarking tool for easily producing benchmarks that compare RAFT's vector search algorithms against other state-of-the-art implementations.
- `raft-ann-bench-cpu` (optional) Reproducible benchmarking tool similar to above, but doesn't require CUDA to be installed on the machine. Can be used to test in environments with competitive CPUs.

Use the following command, depending on your CUDA version, to install all of the RAFT packages with conda (replace `rapidsai` with `rapidsai-nightly` to install more up-to-date but less stable nightly packages). `mamba` is preferred over the `conda` command.
```bash
# for CUDA 11.8
mamba install -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft cuda-version=11.8
```

```bash
# for CUDA 12.0
mamba install -c rapidsai -c conda-forge -c nvidia raft-dask pylibraft cuda-version=12.0
```

Note that the above commands will also install `libraft-headers` and `libraft`.

You can also install the conda packages individually using the `mamba` command above. For example, if you'd like to install RAFT's headers and pre-compiled shared library to use in your project:
```bash
# for CUDA 12.0
mamba install -c rapidsai -c conda-forge -c nvidia libraft libraft-headers cuda-version=12.0
```

If installing the C++ APIs please see [using libraft](https://docs.rapids.ai/api/raft/nightly/using_libraft/) for more information on using the pre-compiled shared library. You can also refer to the [example C++ template project](https://github.com/rapidsai/raft/tree/branch-24.02/cpp/template) for a ready-to-go CMake configuration that you can drop into your project and build against installed RAFT development artifacts above.

### Installing Python through Pip

`pylibraft` and `raft-dask` both have experimental packages that can be [installed through pip](https://rapids.ai/pip.html#install):
```bash
pip install pylibraft-cu11 --extra-index-url=https://pypi.nvidia.com
pip install raft-dask-cu11 --extra-index-url=https://pypi.nvidia.com
```

These packages statically build RAFT's pre-compiled instantiations and so the C++ headers and pre-compiled shared library won't be readily available to use in your code.

The [build instructions](https://docs.rapids.ai/api/raft/nightly/build/) contain more details on building RAFT from source and including it in downstream projects. You can also find a more comprehensive version of the above CPM code snippet the [Building RAFT C++ and Python from source](https://docs.rapids.ai/api/raft/nightly/build/#building-c-and-python-from-source) section of the build instructions.

You can find an example [RAFT project template](cpp/template/README.md) in the `cpp/template` directory, which demonstrates how to build a new application with RAFT or incorporate RAFT into an existing CMake project.


## Contributing

If you are interested in contributing to the RAFT project, please read our [Contributing guidelines](docs/source/contributing.md). Refer to the [Developer Guide](docs/source/developer_guide.md) for details on the developer guidelines, workflows, and principals.

## References

When citing RAFT generally, please consider referencing this Github project.
```bibtex
@misc{rapidsai,
  title={Rapidsai/raft: RAFT contains fundamental widely-used algorithms and primitives for data science, Graph and machine learning.},
  url={https://github.com/rapidsai/raft},
  journal={GitHub},
  publisher={Nvidia RAPIDS},
  author={Rapidsai},
  year={2022}
}
```
If citing the sparse pairwise distances API, please consider using the following bibtex:
```bibtex
@article{nolet2021semiring,
  title={Semiring primitives for sparse neighborhood methods on the gpu},
  author={Nolet, Corey J and Gala, Divye and Raff, Edward and Eaton, Joe and Rees, Brad and Zedlewski, John and Oates, Tim},
  journal={arXiv preprint arXiv:2104.06357},
  year={2021}
}
```

If citing the single-linkage agglomerative clustering APIs, please consider the following bibtex:
```bibtex
@misc{nolet2023cuslink,
      title={cuSLINK: Single-linkage Agglomerative Clustering on the GPU},
      author={Corey J. Nolet and Divye Gala and Alex Fender and Mahesh Doijade and Joe Eaton and Edward Raff and John Zedlewski and Brad Rees and Tim Oates},
      year={2023},
      eprint={2306.16354},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

If citing CAGRA, please consider the following bibtex:
```bibtex
@misc{ootomo2023cagra,
      title={CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs},
      author={Hiroyuki Ootomo and Akira Naruse and Corey Nolet and Ray Wang and Tamas Feher and Yong Wang},
      year={2023},
      eprint={2308.15136},
      archivePrefix={arXiv},
      primaryClass={cs.DS}
}
```

If citing the k-selection routines, please consider the following bibtex:

```bibtex
@proceedings{10.1145/3581784,
    title = {SC '23: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
    year = {2023},
    isbn = {9798400701092},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    abstract = {Started in 1988, the SC Conference has become the annual nexus for researchers and practitioners from academia, industry and government to share information and foster collaborations to advance the state of the art in High Performance Computing (HPC), Networking, Storage, and Analysis.},
    location = {, Denver, CO, USA, }
}
```

If citing the nearest neighbors descent API, please consider the following bibtex:
```bibtex
@inproceedings{10.1145/3459637.3482344,
    author = {Wang, Hui and Zhao, Wan-Lei and Zeng, Xiangxiang and Yang, Jianye},
    title = {Fast K-NN Graph Construction by GPU Based NN-Descent},
    year = {2021},
    isbn = {9781450384469},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3459637.3482344},
    doi = {10.1145/3459637.3482344},
    abstract = {NN-Descent is a classic k-NN graph construction approach. It is still widely employed in machine learning, computer vision, and information retrieval tasks due to its efficiency and genericness. However, the current design only works well on CPU. In this paper, NN-Descent has been redesigned to adapt to the GPU architecture. A new graph update strategy called selective update is proposed. It reduces the data exchange between GPU cores and GPU global memory significantly, which is the processing bottleneck under GPU computation architecture. This redesign leads to full exploitation of the parallelism of the GPU hardware. In the meantime, the genericness, as well as the simplicity of NN-Descent, are well-preserved. Moreover, a procedure that allows to k-NN graph to be merged efficiently on GPU is proposed. It makes the construction of high-quality k-NN graphs for out-of-GPU-memory datasets tractable. Our approach is 100-250\texttimes{} faster than the single-thread NN-Descent and is 2.5-5\texttimes{} faster than the existing GPU-based approaches as we tested on million as well as billion scale datasets.},
    booktitle = {Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
    pages = {1929–1938},
    numpages = {10},
    keywords = {high-dimensional, nn-descent, gpu, k-nearest neighbor graph},
    location = {Virtual Event, Queensland, Australia},
    series = {CIKM '21}
}
```