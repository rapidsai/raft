/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define JSON_DIAGNOSTICS 1
#include <nlohmann/json.hpp>

#undef WARP_SIZE
#ifdef RAFT_ANN_BENCH_USE_RAFT_BRUTE_FORCE
#include "raft_wrapper.h"
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
#include "raft_ivf_flat_wrapper.h"
extern template class raft::bench::ann::RaftIvfFlatGpu<float, int64_t>;
extern template class raft::bench::ann::RaftIvfFlatGpu<uint8_t, int64_t>;
extern template class raft::bench::ann::RaftIvfFlatGpu<int8_t, int64_t>;
#endif
#if defined(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ) || defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA) || \
  defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB)
#include "raft_ivf_pq_wrapper.h"
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_PQ
extern template class raft::bench::ann::RaftIvfPQ<float, int64_t>;
extern template class raft::bench::ann::RaftIvfPQ<uint8_t, int64_t>;
extern template class raft::bench::ann::RaftIvfPQ<int8_t, int64_t>;
#endif
#if defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA) || defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB)
#include "raft_cagra_wrapper.h"
#endif
#ifdef RAFT_ANN_BENCH_USE_RAFT_CAGRA
extern template class raft::bench::ann::RaftCagra<float, uint32_t>;
extern template class raft::bench::ann::RaftCagra<uint8_t, uint32_t>;
extern template class raft::bench::ann::RaftCagra<int8_t, uint32_t>;
#endif

#ifdef RAFT_ANN_BENCH_USE_RAFT_IVF_FLAT
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::RaftIvfFlatGpu<T, IdxT>::BuildParam& param)
{
  param.n_lists = conf.at("nlist");
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) { param.kmeans_trainset_fraction = 1.0 / (double)conf.at("ratio"); }
  if (conf.contains("random_seed")) { param.random_seed = conf.at("random_seed"); }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::RaftIvfFlatGpu<T, IdxT>::SearchParam& param)
{
  param.ivf_flat_params.n_probes = conf.at("nprobe");
}
#endif

#if defined(RAFT_ANN_BENCH_USE_RAFT_IVF_PQ) || defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA) || \
  defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB)
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::RaftIvfPQ<T, IdxT>::BuildParam& param)
{
  if (conf.contains("nlist")) { param.n_lists = conf.at("nlist"); }
  if (conf.contains("niter")) { param.kmeans_n_iters = conf.at("niter"); }
  if (conf.contains("ratio")) { param.kmeans_trainset_fraction = 1.0 / (double)conf.at("ratio"); }
  if (conf.contains("pq_bits")) { param.pq_bits = conf.at("pq_bits"); }
  if (conf.contains("pq_dim")) { param.pq_dim = conf.at("pq_dim"); }
  if (conf.contains("codebook_kind")) {
    std::string kind = conf.at("codebook_kind");
    if (kind == "cluster") {
      param.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
    } else if (kind == "subspace") {
      param.codebook_kind = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    } else {
      throw std::runtime_error("codebook_kind: '" + kind +
                               "', should be either 'cluster' or 'subspace'");
    }
  }
  if (conf.contains("random_seed")) { param.random_seed = conf.at("random_seed"); }
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::RaftIvfPQ<T, IdxT>::SearchParam& param)
{
  if (conf.contains("nprobe")) { param.pq_param.n_probes = conf.at("nprobe"); }
  if (conf.contains("internalDistanceDtype")) {
    std::string type = conf.at("internalDistanceDtype");
    if (type == "float") {
      param.pq_param.internal_distance_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.internal_distance_dtype = CUDA_R_16F;
    } else {
      throw std::runtime_error("internalDistanceDtype: '" + type +
                               "', should be either 'float' or 'half'");
    }
  } else {
    // set half as default type
    param.pq_param.internal_distance_dtype = CUDA_R_16F;
  }

  if (conf.contains("smemLutDtype")) {
    std::string type = conf.at("smemLutDtype");
    if (type == "float") {
      param.pq_param.lut_dtype = CUDA_R_32F;
    } else if (type == "half") {
      param.pq_param.lut_dtype = CUDA_R_16F;
    } else if (type == "fp8") {
      param.pq_param.lut_dtype = CUDA_R_8U;
    } else {
      throw std::runtime_error("smemLutDtype: '" + type +
                               "', should be either 'float', 'half' or 'fp8'");
    }
  } else {
    // set half as default
    param.pq_param.lut_dtype = CUDA_R_16F;
  }
  if (conf.contains("refine_ratio")) {
    param.refine_ratio = conf.at("refine_ratio");
    if (param.refine_ratio < 1.0f) { throw std::runtime_error("refine_ratio should be >= 1.0"); }
  }
}
#endif

#if defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA) || defined(RAFT_ANN_BENCH_USE_RAFT_CAGRA_HNSWLIB)
template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       raft::neighbors::experimental::nn_descent::index_params& param)
{
  if (conf.contains("graph_degree")) { param.graph_degree = conf.at("graph_degree"); }
  if (conf.contains("intermediate_graph_degree")) {
    param.intermediate_graph_degree = conf.at("intermediate_graph_degree");
  }
  // we allow niter shorthand for max_iterations
  if (conf.contains("niter")) { param.max_iterations = conf.at("niter"); }
  if (conf.contains("max_iterations")) { param.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("termination_threshold")) {
    param.termination_threshold = conf.at("termination_threshold");
  }
}

nlohmann::json collect_conf_with_prefix(const nlohmann::json& conf,
                                        const std::string& prefix,
                                        bool remove_prefix = true)
{
  nlohmann::json out;
  for (auto& i : conf.items()) {
    if (i.key().compare(0, prefix.size(), prefix) == 0) {
      auto new_key = remove_prefix ? i.key().substr(prefix.size()) : i.key();
      out[new_key] = i.value();
    }
  }
  return out;
}

template <typename T, typename IdxT>
void parse_build_param(const nlohmann::json& conf,
                       typename raft::bench::ann::RaftCagra<T, IdxT>::BuildParam& param)
{
  if (conf.contains("graph_degree")) {
    param.cagra_params.graph_degree              = conf.at("graph_degree");
    param.cagra_params.intermediate_graph_degree = param.cagra_params.graph_degree * 2;
  }
  if (conf.contains("intermediate_graph_degree")) {
    param.cagra_params.intermediate_graph_degree = conf.at("intermediate_graph_degree");
  }
  if (conf.contains("graph_build_algo")) {
    if (conf.at("graph_build_algo") == "IVF_PQ") {
      param.cagra_params.build_algo = raft::neighbors::cagra::graph_build_algo::IVF_PQ;
    } else if (conf.at("graph_build_algo") == "NN_DESCENT") {
      param.cagra_params.build_algo = raft::neighbors::cagra::graph_build_algo::NN_DESCENT;
    }
  }
  nlohmann::json ivf_pq_build_conf = collect_conf_with_prefix(conf, "ivf_pq_build_");
  if (!ivf_pq_build_conf.empty()) {
    raft::neighbors::ivf_pq::index_params bparam;
    parse_build_param<T, IdxT>(ivf_pq_build_conf, bparam);
    param.ivf_pq_build_params = bparam;
  }
  nlohmann::json ivf_pq_search_conf = collect_conf_with_prefix(conf, "ivf_pq_search_");
  if (!ivf_pq_search_conf.empty()) {
    typename raft::bench::ann::RaftIvfPQ<T, IdxT>::SearchParam sparam;
    parse_search_param<T, IdxT>(ivf_pq_search_conf, sparam);
    param.ivf_pq_search_params = sparam.pq_param;
    param.ivf_pq_refine_rate   = sparam.refine_ratio;
  }
  nlohmann::json nn_descent_conf = collect_conf_with_prefix(conf, "nn_descent_");
  if (!nn_descent_conf.empty()) {
    raft::neighbors::experimental::nn_descent::index_params nn_param;
    nn_param.intermediate_graph_degree = 1.5 * param.cagra_params.intermediate_graph_degree;
    parse_build_param<T, IdxT>(nn_descent_conf, nn_param);
    if (nn_param.graph_degree != param.cagra_params.intermediate_graph_degree) {
      nn_param.graph_degree = param.cagra_params.intermediate_graph_degree;
    }
    param.nn_descent_params = nn_param;
  }
}

raft::bench::ann::AllocatorType parse_allocator(std::string mem_type)
{
  if (mem_type == "device") {
    return raft::bench::ann::AllocatorType::Device;
  } else if (mem_type == "host_pinned") {
    return raft::bench::ann::AllocatorType::HostPinned;
  } else if (mem_type == "host_huge_page") {
    return raft::bench::ann::AllocatorType::HostHugePage;
  }
  THROW(
    "Invalid value for memory type %s, must be one of [\"device\", \"host_pinned\", "
    "\"host_huge_page\"",
    mem_type.c_str());
}

template <typename T, typename IdxT>
void parse_search_param(const nlohmann::json& conf,
                        typename raft::bench::ann::RaftCagra<T, IdxT>::SearchParam& param)
{
  if (conf.contains("itopk")) { param.p.itopk_size = conf.at("itopk"); }
  if (conf.contains("search_width")) { param.p.search_width = conf.at("search_width"); }
  if (conf.contains("max_iterations")) { param.p.max_iterations = conf.at("max_iterations"); }
  if (conf.contains("algo")) {
    if (conf.at("algo") == "single_cta") {
      param.p.algo = raft::neighbors::experimental::cagra::search_algo::SINGLE_CTA;
    } else if (conf.at("algo") == "multi_cta") {
      param.p.algo = raft::neighbors::experimental::cagra::search_algo::MULTI_CTA;
    } else if (conf.at("algo") == "multi_kernel") {
      param.p.algo = raft::neighbors::experimental::cagra::search_algo::MULTI_KERNEL;
    } else if (conf.at("algo") == "auto") {
      param.p.algo = raft::neighbors::experimental::cagra::search_algo::AUTO;
    } else {
      std::string tmp = conf.at("algo");
      THROW("Invalid value for algo: %s", tmp.c_str());
    }
  }
  if (conf.contains("graph_memory_type")) {
    param.graph_mem = parse_allocator(conf.at("graph_memory_type"));
  }
  if (conf.contains("internal_dataset_memory_type")) {
    param.dataset_mem = parse_allocator(conf.at("internal_dataset_memory_type"));
  }
}
#endif
