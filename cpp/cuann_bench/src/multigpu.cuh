/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#ifndef MULTIGPU_H_
#define MULTIGPU_H_

#include <nccl.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cuivfl/src/topk/warp_sort_topk.cuh>
#include <fstream>
#include <thread>
#include <type_traits>
#include <vector>

#include "ann.h"
#include "cudart_util.h"

#define NCCLCHECK(cmd)                                                                      \
  do {                                                                                      \
    ncclResult_t r = cmd;                                                                   \
    if (r != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                                   \
    }                                                                                       \
  } while (0)

namespace {

__global__ void add_index_offset_kernel(size_t* arr, size_t len, size_t offset)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) arr[id] += offset;
}

template <typename T>
__global__ void reset_search_data_placement_kernel(
  T* arr, T* from, int len, int k, int batch_size, int dev_cnt)
{
  size_t id       = blockIdx.x * blockDim.x + threadIdx.x;
  size_t dev      = id / (k * batch_size);
  size_t batch_id = id % (k * batch_size) / k;
  size_t to_id    = batch_id * k * dev_cnt + dev * k + id % k;
  if (id < len) arr[to_id] = from[id];
}

template <typename T>
constexpr ncclDataType_t get_nccl_datatype()
{
  if (std::is_same_v<T, float>) {
    static_assert(sizeof(float) == 4, "float size is not 32 bits");
    return ncclFloat32;
  }
  if (std::is_same_v<T, uint64_t>) return ncclUint64;
  if (std::is_same_v<T, int8_t>) return ncclInt8;
  if (std::is_same_v<T, uint8_t>) return ncclUint8;
  if (std::is_same_v<T, __half>) return ncclFloat16;
  throw std::runtime_error("no supported nccl datatype");
}

class DeviceRestorer {
 public:
  DeviceRestorer() { ANN_CUDA_CHECK(cudaGetDevice(&cur_dev)); }
  ~DeviceRestorer() { ANN_CUDA_CHECK(cudaSetDevice(cur_dev)); }

 private:
  int cur_dev;
};

}  // namespace

namespace cuann {

template <typename T, typename Algo>
class MultiGpuANN : public ANN<T> {
 public:
  using typename ANN<T>::AnnSearchParam;

  MultiGpuANN(Metric metric,
              int dim,
              const typename Algo::BuildParam& param,
              const std::vector<int>& dev_list);

  ~MultiGpuANN();

  void build(const T* dataset, size_t nrow, cudaStream_t stream = 0) override;

  void set_search_param(const AnnSearchParam& param) override;

  void search(const T* queries,
              int batch_size,
              int k,
              size_t* neighbors,
              float* distances,
              cudaStream_t stream = 0) const override;

  void save(const std::string& file) const override;
  void load(const std::string& file) override;

  AlgoProperty get_property() const override
  {
    AlgoProperty property;
    if (dev_ann_property_.dataset_memory_type == MemoryType::Host) {
      property.dataset_memory_type = MemoryType::Host;
    } else if (dev_ann_property_.dataset_memory_type == MemoryType::Device) {
      property.dataset_memory_type = MemoryType::HostMmap;
    } else {
      throw std::runtime_error("multigpu: invalid device algo dataset memory type");
    }
    property.query_memory_type        = MemoryType::Device;
    property.need_dataset_when_search = dev_ann_property_.need_dataset_when_search;
    return property;
  }

  void set_search_dataset(const T* dataset, size_t nrow) override;

 private:
  void distribute_dataset_(const T* dataset, size_t nrow);
  void add_index_offset_(size_t* arr, size_t len, size_t offset, cudaStream_t stream) const;
  void set_wait_for_all_streams_(cudaStream_t stream) const;
  template <typename U>
  void reset_search_data_placement_(
    U* arr, U* from, int k, int batch_size, size_t all_result_size, cudaStream_t stream) const;

  const static int block_size_ = 256;
  using ANN<T>::dim_;
  std::vector<cudaEvent_t> event_;
  std::vector<std::unique_ptr<Algo>> dev_ann_interface_;
  AlgoProperty dev_ann_property_;
  std::vector<int> dev_id_;
  std::vector<T*> d_data_;
  std::vector<cudaStream_t> dev_stream_;
  std::vector<cudaMemPool_t> mempool_;
  std::vector<ncclComm_t> comms_;
  std::vector<size_t> dev_data_offset_;
  int dev_cnt_;
  size_t nrow_;
};

template <typename T, typename Algo>
MultiGpuANN<T, Algo>::MultiGpuANN(Metric metric,
                                  int dim,
                                  const typename Algo::BuildParam& param,
                                  const std::vector<int>& dev_list)
  : ANN<T>(metric, dim),
    dev_cnt_(dev_list.size()),
    dev_ann_interface_(dev_list.size()),
    dev_id_(dev_list),
    d_data_(dev_list.size()),
    dev_stream_(dev_list.size()),
    event_(dev_list.size()),
    mempool_(dev_list.size()),
    comms_(dev_list.size()),
    dev_data_offset_(dev_list.size())
{
  DeviceRestorer restore_dev;
  uint64_t threshold = UINT64_MAX;
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    ANN_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool_[i], dev_id_[i]));
    ANN_CUDA_CHECK(
      cudaMemPoolSetAttribute(mempool_[i], cudaMemPoolAttrReleaseThreshold, &threshold));
    std::vector<cudaMemAccessDesc> desc;
    for (int j = 0; j < dev_cnt_; j++) {
      if (i == j) continue;
      cudaMemAccessDesc tmp_desc;
      tmp_desc.location.type = cudaMemLocationTypeDevice;
      tmp_desc.location.id   = dev_id_[j];
      tmp_desc.flags         = cudaMemAccessFlagsProtReadWrite;
      desc.push_back(tmp_desc);
    }
    ANN_CUDA_CHECK(cudaMemPoolSetAccess(mempool_[i], desc.data(), desc.size()));
    ANN_CUDA_CHECK(cudaStreamCreate(&dev_stream_[i]));
    ANN_CUDA_CHECK(cudaEventCreate(&event_[i], cudaEventDisableTiming));
    dev_ann_interface_[i] = std::make_unique<Algo>(metric, dim, param);
  }
  NCCLCHECK(ncclCommInitAll(comms_.data(), dev_cnt_, dev_id_.data()));

  dev_ann_property_ = dev_ann_interface_[0]->get_property();
  if (dev_ann_property_.query_memory_type != MemoryType::Device) {
    throw std::runtime_error("multigpu: query_memory_type of dev_algo must be DEVICE!");
  }
}

template <typename T, typename Algo>
MultiGpuANN<T, Algo>::~MultiGpuANN()
{
  DeviceRestorer restore_dev;
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    if (d_data_[i] && dev_ann_property_.dataset_memory_type == MemoryType::Device) {
      ANN_CUDA_CHECK(cudaFree(d_data_[i]));
    }
    ANN_CUDA_CHECK(cudaStreamDestroy(dev_stream_[i]));
    ANN_CUDA_CHECK(cudaEventDestroy(event_[i]));
    NCCLCHECK(ncclCommDestroy(comms_[i]));
  }
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::build(const T* dataset, size_t nrow, cudaStream_t stream)
{
  DeviceRestorer restore_dev;
  distribute_dataset_(dataset, nrow);
  nrow_ = nrow;

  std::vector<std::thread> threads;

  size_t basic_size = nrow / dev_cnt_;
  size_t offset     = 0;
  int mod           = nrow % dev_cnt_;
  for (int i = 0; i < dev_cnt_; i++) {
    size_t data_size = basic_size + (mod > i ? 1 : 0);
    threads.emplace_back([&, i, data_size]() {
      ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
      dev_ann_interface_[i]->build(d_data_[i], data_size, dev_stream_[i]);
    });
    dev_data_offset_[i] = offset;
    offset += data_size;
  }
  for (auto& it : threads)
    it.join();

  set_wait_for_all_streams_(stream);
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::set_search_param(const AnnSearchParam& param)
{
  DeviceRestorer restore_dev;
  auto search_param = dynamic_cast<const typename Algo::SearchParam&>(param);
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    dev_ann_interface_[i]->set_search_param(search_param);
  }
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::search(const T* queries,
                                  int batch_size,
                                  int k,
                                  size_t* neighbors,
                                  float* distances,
                                  cudaStream_t stream) const
{
  DeviceRestorer restore_dev;

  std::vector<T*> d_queries(dev_cnt_);
  std::vector<size_t*> d_neighbors(dev_cnt_);
  std::vector<float*> d_distances(dev_cnt_);

  float* candidate_distances;
  float* result_distances;
  size_t* candidate_neighbors;
  size_t* result_neighbors;

  int cur_dev;
  ANN_CUDA_CHECK(cudaGetDevice(&cur_dev));

  auto cur_dev_it = std::find(dev_id_.begin(), dev_id_.end(), cur_dev);
  if (cur_dev_it == dev_id_.end()) {
    throw std::runtime_error("current device is not in dev_list!");
  }
  int cur_dev_id = cur_dev_it - dev_id_.begin();

  size_t single_dev_result_size = static_cast<size_t>(k) * batch_size;
  size_t all_result_size        = single_dev_result_size * dev_cnt_;

  ANN_CUDA_CHECK(cudaMallocAsync(
    &candidate_distances, all_result_size * sizeof(float), dev_stream_[cur_dev_id]));
  ANN_CUDA_CHECK(cudaMallocAsync(
    &candidate_neighbors, all_result_size * sizeof(size_t), dev_stream_[cur_dev_id]));
  ANN_CUDA_CHECK(
    cudaMallocAsync(&result_distances, all_result_size * sizeof(float), dev_stream_[cur_dev_id]));
  ANN_CUDA_CHECK(
    cudaMallocAsync(&result_neighbors, all_result_size * sizeof(size_t), dev_stream_[cur_dev_id]));

  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    ANN_CUDA_CHECK(cudaMallocAsync(&d_queries[i], batch_size * dim_ * sizeof(T), dev_stream_[i]));
    ANN_CUDA_CHECK(
      cudaMallocAsync(&d_neighbors[i], single_dev_result_size * sizeof(size_t), dev_stream_[i]));
    ANN_CUDA_CHECK(
      cudaMallocAsync(&d_distances[i], single_dev_result_size * sizeof(float), dev_stream_[i]));
  }
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < dev_cnt_; i++) {
    NCCLCHECK(ncclBroadcast(queries,
                            d_queries[i],
                            batch_size * dim_,
                            get_nccl_datatype<T>(),
                            cur_dev_id,
                            comms_[i],
                            dev_stream_[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  std::vector<std::thread> threads;

  for (int i = 0; i < dev_cnt_; i++) {
    threads.emplace_back([&, i]() {
      ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
      dev_ann_interface_[i]->search(
        d_queries[i], batch_size, k, d_neighbors[i], d_distances[i], dev_stream_[i]);
      add_index_offset_(
        d_neighbors[i], single_dev_result_size, dev_data_offset_[i], dev_stream_[i]);
    });
  }

  for (auto& it : threads)
    it.join();

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < dev_cnt_; i++) {
    NCCLCHECK(ncclRecv(result_distances + i * single_dev_result_size,
                       single_dev_result_size,
                       get_nccl_datatype<float>(),
                       i,
                       comms_[cur_dev_id],
                       dev_stream_[cur_dev_id]));
    NCCLCHECK(ncclRecv(result_neighbors + i * single_dev_result_size,
                       single_dev_result_size,
                       get_nccl_datatype<size_t>(),
                       i,
                       comms_[cur_dev_id],
                       dev_stream_[cur_dev_id]));
  }
  for (int i = 0; i < dev_cnt_; i++) {
    NCCLCHECK(ncclSend(d_distances[i],
                       single_dev_result_size,
                       get_nccl_datatype<float>(),
                       cur_dev_id,
                       comms_[i],
                       dev_stream_[i]));
    NCCLCHECK(ncclSend(d_neighbors[i],
                       single_dev_result_size,
                       get_nccl_datatype<size_t>(),
                       cur_dev_id,
                       comms_[i],
                       dev_stream_[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  set_wait_for_all_streams_(stream);

  ANN_CUDA_CHECK(cudaSetDevice(dev_id_[cur_dev_id]));

  reset_search_data_placement_(
    candidate_distances, result_distances, k, batch_size, all_result_size, stream);
  reset_search_data_placement_(
    candidate_neighbors, result_neighbors, k, batch_size, all_result_size, stream);

  void* warp_sort_topk_buf = nullptr;
  size_t buf_size          = 0;

  nv::warp_sort_topk<float, size_t>(nullptr,
                                    buf_size,
                                    candidate_distances,
                                    candidate_neighbors,
                                    batch_size,
                                    k * dev_cnt_,
                                    k,
                                    distances,
                                    neighbors,
                                    false,
                                    stream);
  ANN_CUDA_CHECK(cudaMallocAsync(&warp_sort_topk_buf, buf_size, stream));
  nv::warp_sort_topk<float, size_t>(warp_sort_topk_buf,
                                    buf_size,
                                    candidate_distances,
                                    candidate_neighbors,
                                    batch_size,
                                    k * dev_cnt_,
                                    k,
                                    distances,
                                    neighbors,
                                    false,
                                    stream);

  ANN_CUDA_CHECK(cudaFreeAsync(warp_sort_topk_buf, stream));
  ANN_CUDA_CHECK(cudaFreeAsync(candidate_neighbors, stream));
  ANN_CUDA_CHECK(cudaFreeAsync(candidate_distances, stream));
  ANN_CUDA_CHECK(cudaFreeAsync(result_neighbors, stream));
  ANN_CUDA_CHECK(cudaFreeAsync(result_distances, stream));
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    ANN_CUDA_CHECK(cudaFreeAsync(d_queries[i], stream));
    ANN_CUDA_CHECK(cudaFreeAsync(d_neighbors[i], stream));
    ANN_CUDA_CHECK(cudaFreeAsync(d_distances[i], stream));
  }
  ANN_CUDA_CHECK_LAST_ERROR()
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::save(const std::string& file) const
{
  DeviceRestorer restore_dev;
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    dev_ann_interface_[i]->save(file + "_" + std::to_string(i));
  }
  std::ofstream ofs(file);
  if (!ofs) { throw std::runtime_error("can't open index file: " + file); }
  ofs << nrow_ << '\n';
  for (auto it : dev_data_offset_)
    ofs << it << '\n';
  ofs.close();
  if (!ofs) { throw std::runtime_error("can't write to index file: " + file); }
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::load(const std::string& file)
{
  DeviceRestorer restore_dev;
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    dev_ann_interface_[i]->load(file + "_" + std::to_string(i));
  }
  std::ifstream ifs(file);
  if (!ifs) { throw std::runtime_error("can't open index file: " + file); }
  ifs >> nrow_;
  for (auto& it : dev_data_offset_)
    ifs >> it;
  ifs.close();
  if (!ifs) { throw std::runtime_error("can't read from index file: " + file); }
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::set_search_dataset(const T* dataset, size_t nrow)
{
  DeviceRestorer restore_dev;
  distribute_dataset_(dataset, nrow);
  size_t basic_size = nrow / dev_cnt_;
  size_t offset     = 0;
  int mod           = nrow % dev_cnt_;
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    size_t data_size = basic_size + (mod > i ? 1 : 0);
    dev_ann_interface_[i]->set_search_dataset(d_data_[i], data_size);
    offset += data_size;
  }
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::distribute_dataset_(const T* dataset, size_t nrow)
{
  size_t basic_size = nrow / dev_cnt_;
  size_t offset     = 0;
  int mod           = nrow % dev_cnt_;
  for (int i = 0; i < dev_cnt_; i++) {
    size_t data_size = (basic_size + (mod > i ? 1 : 0)) * dim_;
    if (dev_ann_property_.dataset_memory_type == MemoryType::Device) {
      ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
      ANN_CUDA_CHECK(cudaMalloc(&d_data_[i], data_size * sizeof(T)));
      ANN_CUDA_CHECK(cudaMemcpyAsync(d_data_[i],
                                     dataset + offset,
                                     data_size * sizeof(T),
                                     cudaMemcpyHostToDevice,
                                     dev_stream_[i]));
    } else {
      d_data_[i] = const_cast<T*>(dataset) + offset;
    }
    offset += data_size;
  }
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::add_index_offset_(size_t* arr,
                                             size_t len,
                                             size_t offset,
                                             cudaStream_t stream) const
{
  add_index_offset_kernel<<<(len + block_size_ - 1) / block_size_, block_size_, 0, stream>>>(
    arr, len, offset);
}

template <typename T, typename Algo>
void MultiGpuANN<T, Algo>::set_wait_for_all_streams_(cudaStream_t stream) const
{
  for (int i = 0; i < dev_cnt_; i++) {
    ANN_CUDA_CHECK(cudaSetDevice(dev_id_[i]));
    ANN_CUDA_CHECK(cudaEventRecord(event_[i], dev_stream_[i]));
    ANN_CUDA_CHECK(cudaStreamWaitEvent(stream, event_[i], 0));
  }
}

template <typename T, typename Algo>
template <typename U>
void MultiGpuANN<T, Algo>::reset_search_data_placement_(
  U* arr, U* from, int k, int batch_size, size_t all_result_size, cudaStream_t stream) const
{
  reset_search_data_placement_kernel<<<(all_result_size + block_size_ - 1) / block_size_,
                                       block_size_,
                                       0,
                                       stream>>>(
    arr, from, all_result_size, k, batch_size, dev_cnt_);
}

}  // namespace cuann

#endif
