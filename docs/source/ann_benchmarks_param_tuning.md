# ANN Benchmarks Parameter Tuning Guide

This guide outlines the various parameter settings that can be specified in [RAFT ANN Benchmark](raft_ann_benchmarks.md) json configuration files and explains the impact they have on corresponding algorithms to help inform their settings for benchmarking across desired levels of recall. 


## RAFT Indexes

### `raft_ivf_flat`

IVF-flat uses an inverted-file index, which partitions the vectors into a series of clusters, or lists, storing them in an interleaved format which is optimized for fast distance computation. The searching of an IVF-flat index reduces the total vectors in the index to those within some user-specified nearest clusters called probes.

IVF-flat is a simple algorithm which won't save any space, but it provides competitive search times even at higher levels of recall.

| Parameter | Type                                  | Data Type             | Description                                                                                                                                                                       |
|-----------|---------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`  | `build_param`                         | Positive Integer `>0` | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `ratio`   | `build_param`                         | Positive Float `>0`   | Fraction of the number of training points which should be used to train the clusters.                                                                                             |
| `nprobe`  | `search_params` | Positive Integer `>0` | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                       |


### `raft_ivf_pq`

IVF-pq is an inverted-file index, which partitions the vectors into a series of clusters, or lists, in a similar way to IVF-flat above. The difference is that IVF-PQ uses product quantization to also compress the vectors, giving the index a smaller memory footprint. Unfortunately, higher levels of compression can also shrink recall, which a refinement step can improve when the original vectors are still available.


| Parameter               | Type           | Data Type                        | Description                                                                                                                                                                       |
|-------------------------|----------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`                | `build_param`  | Positive Integer `>0`            | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `niter`                 | `build_param`  | Positive Integer `>0`            | Number of k-means iterations to use when training the clusters.                                                                                                                   |
| `pq_dim`                | `build_param`  | Positive Integer. Multiple of 8. | Dimensionality of the vector after product quantization. When 0, a heuristic is used to select this value. `pq_dim` * `pq_bits` must be a multiple of 8.                          |
| `pq_bits`               | `build_param`  | Positive Integer. `4-8`          | Bit length of the vector element after quantization.                                                                                                                              |
| `nprobe`                | `search_params` | Positive Integer `>0`            | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index.                                       |
| `internalDistanceDtype` | `search_params` | [`float`, `half`]                | The precision to use for the distance computations. Lower precision can increase performance at the cost of accuracy.                                                             |
| `smemLutDtype`          | `search_params` | [`float`, `half`, `fp8`]         | The precision to use for the lookup table in shared memory. Lower precision can increase performance at the cost of accuracy.                                                     |
| `refine_ratio`          | `search_params` | Positive Number `>=0`             | `refine_ratio * k` nearest neighbors are queried from the index initially and an additional refinement step improves recall by selecting only the best `k` neighbors. .           |


### `raft_cagra`


## FAISS Indexes

### `faiss_gpu_ivf_flat`

IVF-flat uses an inverted-file index, which partitions the vectors into a series of clusters, or lists, storing them in an interleaved format which is optimized for fast distance computation. The searching of an IVF-flat index reduces the total vectors in the index to those within some user-specified nearest clusters called probes.

IVF-flat is a simple algorithm which won't save any space, but it provides competitive search times even at higher levels of recall.

| Parameter | Type           | Data Type             | Description                                                                                                                                                                       |
|-----------|----------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`  | `build_param`  | Positive Integer `>0` | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `nprobe`  | `search_params` | Positive Integer `>0` | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |

### `faiss_gpu_ivf_pq`

IVF-pq is an inverted-file index, which partitions the vectors into a series of clusters, or lists, in a similar way to IVF-flat above. The difference is that IVF-PQ uses product quantization to also compress the vectors, giving the index a smaller memory footprint. Unfortunately, higher levels of compression can also shrink recall, which a refinement step can improve when the original vectors are still available.

| Parameter | Type                                  | Data Type             | Description                                                                                                                                                                       |
|-----------|---------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`  | `build_param`                         | Positive Integer `>0` | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `nprobe`  | `search_params` | Positive Integer `>0` | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |



## HNSW

### `hnswlib`

## GGNN Index

### `ggnn`
