# ANN Benchmarks Parameter Tuning Guide

This guide outlines the various parameter settings that can be specified in [RAFT ANN Benchmark](raft_ann_benchmarks.md) json configuration files and explains the impact they have on corresponding algorithms to help inform their settings for benchmarking across desired levels of recall.


## RAFT Indexes

### `raft_brute_force`

Use RAFT brute-force index for exact search. Brute-force has no further build or search parameters.

### `raft_ivf_flat`

IVF-flat uses an inverted-file index, which partitions the vectors into a series of clusters, or lists, storing them in an interleaved format which is optimized for fast distance computation. The searching of an IVF-flat index reduces the total vectors in the index to those within some user-specified nearest clusters called probes.

IVF-flat is a simple algorithm which won't save any space, but it provides competitive search times even at higher levels of recall.

| Parameter            | Type             | Required | Data Type                  | Default  | Description                                                                                                                                                                       |
|----------------------|------------------|----------|----------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlist`              | `build`    | Y        | Positive Integer >0        |          | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `niter`              | `build`    | N        | Positive Integer >0        | 20       | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ratio`              | `build`    | N        | Positive Integer >0        | 2        | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                            |
| `dataset_memory_type` | `build` | N | ["device", "host", "mmap"] | "mmap" | What memory type should the dataset reside?                                                                                                                                       |
| `query_memory_type`  | `search` | N | ["device", "host", "mmap"] | "device | What memory type should the queries reside? |
| `nprobe`             | `search`  | Y        | Positive Integer >0        |          | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                       |


### `raft_ivf_pq`

IVF-pq is an inverted-file index, which partitions the vectors into a series of clusters, or lists, in a similar way to IVF-flat above. The difference is that IVF-PQ uses product quantization to also compress the vectors, giving the index a smaller memory footprint. Unfortunately, higher levels of compression can also shrink recall, which a refinement step can improve when the original vectors are still available.

| Parameter              | Type           | Required | Data Type                        | Default | Description                                                                                                                                                                     |
|------------------------|----------------|---|----------------------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlist`                | `build`  | Y | Positive Integer >0              |         | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `niter`                | `build`  | N | Positive Integer >0              | 20      | Number of k-means iterations to use when training the clusters.                                                                                                                 |
| `ratio`                | `build`  | N | Positive Integer >0              | 2       | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                            |
| `pq_dim`               | `build`  | N | Positive Integer. Multiple of 8. | 0       | Dimensionality of the vector after product quantization. When 0, a heuristic is used to select this value. `pq_dim` * `pq_bits` must be a multiple of 8.                        |
| `pq_bits`              | `build`  | N | Positive Integer. [4-8]          | 8       | Bit length of the vector element after quantization.                                                                                                                            |
| `codebook_kind`        | `build`  | N | ["cluster", "subspace"]          | "subspace" | Type of codebook. See the [API docs](https://docs.rapids.ai/api/raft/nightly/cpp_api/neighbors_ivf_pq/#_CPPv412codebook_gen) for more detail                                 |
| `dataset_memory_type`  | `build` | N | ["device", "host", "mmap"]       | "host" | What memory type should the dataset reside?                                                                                                                                       |
| `max_train_points_per_pq_code`         | `build` | N | Positive Number >=1              | 256       | Max number of data points per PQ code used for PQ code book creation. Depending on input dataset size, the data points could be less than what user specifies.         |
| `query_memory_type`    | `search` | N | ["device", "host", "mmap"]       | "device | What memory type should the queries reside? |
| `nprobe`               | `search` | Y | Positive Integer >0              |         | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                     |
| `internalDistanceDtype` | `search` | N | [`float`, `half`]                | `half`  | The precision to use for the distance computations. Lower precision can increase performance at the cost of accuracy.                                                           |
| `smemLutDtype`         | `search` | N | [`float`, `half`, `fp8`]         | `half`  | The precision to use for the lookup table in shared memory. Lower precision can increase performance at the cost of accuracy.                                                   |
| `refine_ratio`         | `search` | N| Positive Number >=1              | 1       | `refine_ratio * k` nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best `k` neighbors.           |


### `raft_cagra`
<a id='raft-cagra'></a>CAGRA uses a graph-based index, which creates an intermediate, approximate kNN graph using IVF-PQ and then further refining and optimizing to create a final kNN graph. This kNN graph is used by CAGRA as an index for search.

| Parameter                   | Type           | Required | Data Type                  | Default | Description                                                                                                                                                                       |
|-----------------------------|----------------|----------|----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `graph_degree`              | `build`  | N        | Positive Integer >0        | 64 | Degree of the final kNN graph index. |
| `intermediate_graph_degree` | `build`  | N        | Positive Integer >0        | 128 | Degree of the intermediate kNN graph. |
| `graph_build_algo`          | `build`  | N | ["IVF_PQ", "NN_DESCENT"]   | "IVF_PQ" | Algorithm to use for search |
| `dataset_memory_type`       | `build`  | N | ["device", "host", "mmap"] | "mmap" | What memory type should the dataset reside while constructing the index?                                                                                                                                       |
| `query_memory_type`         | `search` | N | ["device", "host", "mmap"] | "device | What memory type should the queries reside? |
| `itopk`                     | `search_wdith`  | N        | Positive Integer >0        | 64 | Number of intermediate search results retained during the search. Higher values improve search accuracy at the cost of speed. |
| `search_width`              | `search`  | N        | Positive Integer >0        | 1 | Number of graph nodes to select as the starting point for the search in each iteration. |
| `max_iterations`            | `search`  | N        | Integer >=0                | 0 | Upper limit of search iterations. Auto select when 0. |
| `algo`                      | `search`  | N        | string                     | "auto" | Algorithm to use for search. Possible values: {"auto", "single_cta", "multi_cta", "multi_kernel"} |
| `graph_memory_type`         | `search`  | N        | string                     | "device" | Memory type to store gaph. Must be one of {"device", "host_pinned", "host_huge_page"}. |
| `internal_dataset_memory_type` | `search`  | N        | string                     | "device" | Memory type to store dataset in the index. Must be one of {"device", "host_pinned", "host_huge_page"}. |

The `graph_memory_type` or `internal_dataset_memory_type` options can be useful for large datasets that do not fit the device memory. Setting `internal_dataset_memory_type` other than `device` has negative impact on search speed. Using `host_huge_page` option is only supported on systems with Heterogeneous Memory Management or on platforms that natively support GPU access to system allocated memory, for example Grace Hopper.

To fine tune CAGRA index building we can customize IVF-PQ index builder options using the following settings. These take effect only if `graph_build_algo == "IVF_PQ"`. It is recommended to experiment using a separate IVF-PQ index to find the config that gives the largest QPS for large batch. Recall does not need to be very high, since CAGRA further optimizes the kNN neighbor graph. Some of the default values are derived from the dataset size which is assumed to be [n_vecs, dim].

| Parameter              | Type           | Required | Data Type                        | Default | Description                                                                                                                                                                     |
|------------------------|----------------|---|----------------------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ivf_pq_build_nlist`                | `build`  | N | Positive Integer >0              | n_vecs / 2500        | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ivf_pq_build_niter`                | `build`  | N | Positive Integer >0              | 25      | Number of k-means iterations to use when training the clusters.                                                                                                                 |
| `ivf_pq_build_ratio`                | `build`  | N | Positive Integer >0              | 10      | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                            |
| `ivf_pq_build_pq_dim`               | `build`  | N | Positive Integer. Multiple of 8. | dim/2 rounded up to 8     | Dimensionality of the vector after product quantization. When 0, a heuristic is used to select this value. `pq_dim` * `pq_bits` must be a multiple of 8.                        |
| `ivf_pq_build_pq_bits`              | `build`  | N | Positive Integer. [4-8]          | 8       | Bit length of the vector element after quantization.                                                                                                                            |
| `ivf_pq_build_codebook_kind`        | `build`  | N | ["cluster", "subspace"]          | "subspace" | Type of codebook. See the [API docs](https://docs.rapids.ai/api/raft/nightly/cpp_api/neighbors_ivf_pq/#_CPPv412codebook_gen) for more detail                                 |
| `ivf_pq_search_nprobe`               | `build` | N | Positive Integer >0              | min(2*dim, nlist)        | The closest number of clusters to search for each query vector.                                    |
| `ivf_pq_search_internalDistanceDtype` | `build` | N | [`float`, `half`]                | `fp8`  | The precision to use for the distance computations. Lower precision can increase performance at the cost of accuracy.                                                           |
| `ivf_pq_search_smemLutDtype`         | `build` | N | [`float`, `half`, `fp8`]         | `half`  | The precision to use for the lookup table in shared memory. Lower precision can increase performance at the cost of accuracy.                                                   |
| `ivf_pq_search_refine_ratio`         | `build` | N| Positive Number >=1              | 2       | `refine_ratio * k` nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best `k` neighbors.           |

Alternatively, if `graph_build_algo == "NN_DESCENT"`, then we can customize the following parameters

| Parameter                   | Type           | Required | Data Type                  | Default | Description                                                                                                                                                                       |
|-----------------------------|----------------|----------|----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nn_descent_niter`          | `build`  | N        | Positive Integer>0         | 20 | Number of NN Descent iterations. |
| `nn_descent_intermediate_graph_degree`          | `build`  | N        | Positive Integer>0         | `intermediate_graph_degree` * 1.5 | Intermadiate graph degree during NN descent iterations |
| `nn_descent_max_iterations`          | `build`  | N        | Positive Integer>0         | 20 | Alias for `nn_descent_niter` |
| `nn_descent_termination_threshold`          | `build`  | N        | Positive float>0         | 0.0001 | Termination threshold for NN descent. |

### `raft_cagra_hnswlib`
This is a benchmark that enables interoperability between `CAGRA` built `HNSW` search. It uses the `CAGRA` built graph as the base layer of an `hnswlib` index to search queries only within the base layer (this is enabled with a simple patch to `hnswlib`).

`build` : Same as `build` of [CAGRA](#raft-cagra)

`search` : Same as `search` of [hnswlib](#hnswlib)

## FAISS Indexes

### `faiss_gpu_flat`

Use FAISS flat index on the GPU, which performs an exact search using brute-force and doesn't have any further build or search parameters. 

### `faiss_gpu_ivf_flat`

IVF-flat uses an inverted-file index, which partitions the vectors into a series of clusters, or lists, storing them in an interleaved format which is optimized for fast distance computation. The searching of an IVF-flat index reduces the total vectors in the index to those within some user-specified nearest clusters called probes.

IVF-flat is a simple algorithm which won't save any space, but it provides competitive search times even at higher levels of recall.

| Parameter | Type           | Required | Data Type           | Default | Description                                                                                                                                                                       |
|-----------|----------------|----------|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`  | `build`  | Y        | Positive Integer >0 |         | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ratio`   | `build`  | N        | Positive Integer >0 | 2       | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                            |
| `nprobe`  | `search` | Y        | Positive Integer >0 | | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                       |

### `faiss_gpu_ivf_pq`

IVF-pq is an inverted-file index, which partitions the vectors into a series of clusters, or lists, in a similar way to IVF-flat above. The difference is that IVF-PQ uses product quantization to also compress the vectors, giving the index a smaller memory footprint. Unfortunately, higher levels of compression can also shrink recall, which a refinement step can improve when the original vectors are still available.

| Parameter        | Type           | Required | Data Type                        | Default | Description                                                                                                                                                                       |
|------------------|----------------|----------|----------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlist`          | `build`  | Y        | Positive Integer >0              |         | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ratio`          | `build`  | N        | Positive Integer >0              | 2       | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                            |
| `M_ratio`        | `build`  | Y        | Positive Integer Power of 2 [8-64] |         | Ratio of numbeer of chunks or subquantizers for each vector. Computed by `dims` / `M_ratio`                                                                                         |
| `usePrecomputed` | `build`  | N        | Boolean. Default=`false`         | `false` | Use pre-computed lookup tables to speed up search at the cost of increased memory usage.                                                                                          |
| `useFloat16`     | `build`  | N        | Boolean. Default=`false`         | `false`  | Use half-precision floats for clustering step.                                                                                                                                    |
| `nprobe`         | `search` | Y        | Positive Integer >0              |         | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                       |
| `refine_ratio`   | `search` | N| Positive Number >=1          | 1       | `refine_ratio * k` nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best `k` neighbors.             |

### `faiss_cpu_flat`

Use FAISS flat index on the CPU, which performs an exact search using brute-force and doesn't have any further build or search parameters.


| Parameter | Type           | Required | Data Type           | Default | Description                                                                                                                                                                       |
|-----------|----------------|----------|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `numThreads`     | `search` | N        | Positive Integer >0                  | 1       | Number of threads to use for queries.                                                                                                                                                                                                                                                             |

### `faiss_cpu_ivf_flat`

Use FAISS IVF-Flat index on CPU

| Parameter | Type           | Required | Data Type           | Default | Description                                                                                                                                                                       |
|----------|----------------|----------|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlist`  | `build`  | Y        | Positive Integer >0 |         | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ratio`  | `build`  | N        | Positive Integer >0 | 2       | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                            |
| `nprobe` | `search` | Y        | Positive Integer >0 | | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                       |
| `numThreads`    | `search` | N        | Positive Integer >0                  | 1       | Number of threads to use for queries.                                                                                                                                                                                                                                                             |

### `faiss_cpu_ivf_pq`

Use FAISS IVF-PQ index on CPU

| Parameter        | Type           | Required | Data Type                          | Default | Description                                                                                                                                                                   |
|------------------|----------------|----------|------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlist`          | `build`  | Y        | Positive Integer >0                |         | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ratio`          | `build`  | N        | Positive Integer >0                | 2       | `1/ratio` is the number of training points which should be used to train the clusters.                                                                                        |
| `M`              | `build`  | Y        | Positive Integer Power of 2 [8-64] |         | Number of chunks or subquantizers for each vector.                                                                                                                            |
| `usePrecomputed` | `build`  | N        | Boolean. Default=`false`           | `false` | Use pre-computed lookup tables to speed up search at the cost of increased memory usage.                                                                                      |
| `bitsPerCode`    | `build`  | N        | Positive Integer [4-8]             | 8       | Number of bits to use for each code.                                                                                                                                          |
| `nprobe`         | `search` | Y        | Positive Integer >0                |         | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                   |
| `refine_ratio`   | `search` | N| Positive Number >=1                | 1       | `refine_ratio * k` nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best `k` neighbors.         |
| `numThreads`     | `search` | N        | Positive Integer >0                  | 1       | Number of threads to use for queries.                                                                                                                                                                                                                                                             |


## HNSW
<a id='hnswlib'></a>
### `hnswlib`

| Parameter        | Type      | Required | Data Type                            | Default | Description                                                                                                                                                                                                                                                                                       |
|------------------|-----------|----------|--------------------------------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `efConstruction` | `build`   | Y        | Positive Integer >0                  |         | Controls index time and accuracy. Bigger values increase the index quality. At some point, increasing this will no longer improve the quality.                                                                                                                                                    |
| `M`              | `build`   | Y        | Positive Integer often between 2-100 |         | Number of bi-directional links create for every new element during construction. Higher values work for higher intrinsic dimensionality and/or high recall, low values can work for datasets with low intrinsic dimensionality and/or low recalls. Also affects the algorithm's memory consumption. |
| `numThreads`     | `build`   | N        | Positive Integer >0                  | 1       | Number of threads to use to build the index.                                                                                                                                                                                                                                                      |
| `ef`             | `search`  | Y        | Positive Integer >0                  |         | Size of the dynamic list for the nearest neighbors used for search. Higher value leads to more accurate but slower search. Cannot be lower than `k`.                                                                                                                                              |
| `numThreads`     | `search` | N        | Positive Integer >0                  | 1       | Number of threads to use for queries.                                                                                                                                                                                                                                                             |

Please refer to [HNSW algorithm parameters guide](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) from `hnswlib` to learn more about these arguments.