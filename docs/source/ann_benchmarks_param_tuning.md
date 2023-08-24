# ANN Benchmarks Parameter Tuning Guide

This guide outlines the various parameter settings that can be specified in [RAFT ANN Benchmark](raft_ann_benchmarks.md) json configuration files and explains the impact they have on corresponding algorithms to help inform their settings for benchmarking across desired levels of recall. 


| Algorithm           | Parameter Options                            |
|---------------------|----------------------------------------------|
| `faiss_gpu_ivf_flat` | `{  }`                                       | `faiss_gpu_ivf_flat`, `faiss_gpu_ivf_pq` |
| GGNN                | `ggnn`                                       |
| HNSWlib             | `hnswlib`                                    |
| RAFT                | `raft_cagra`, `raft_ivf_flat`, `raft_ivf_pq` |


