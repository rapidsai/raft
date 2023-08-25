# ANN Benchmarks Parameter Tuning Guide

This guide outlines the various parameter settings that can be specified in [RAFT ANN Benchmark](raft_ann_benchmarks.md) json configuration files and explains the impact they have on corresponding algorithms to help inform their settings for benchmarking across desired levels of recall. 


## RAFT Indexes

### IVF-Flat

| Parameter | Type           | Data Type             | Description                                                                                                                                                                       |
|-----------|----------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`  | `build_param`  | Positive Integer `>0` | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `nprobe`  | `search_params` | Positive Integer `>0` | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |

### IVF-PQ

| Parameter | Type                                  | Data Type             | Description                                                                                                                                                                       |
|-----------|---------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `nlists`  | `build_param`                         | Positive Integer `>0` | Number of clusters to partition the vectors into. Larger values will put less points into each cluster but this will impact index build time as more clusters need to be trained. |
| `nprobe`  | `search_params` | Positive Integer `>0` | The closest number of clusters to search for each query vector. Larger values will improve recall but will search more points in the index. |


## FAISS Indexes

### IVF-Flat

### IVF-PQ


## HNSWLib Index


## GGNN Index