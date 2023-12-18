
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

#include <cstdint>
#include <raft/core/c_api.h>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <raft/core/interop.cuh>
#include <raft/neighbors/cagra.h>
#include <raft/neighbors/cagra_types.hpp>
#include <raft_runtime/neighbors/cagra.hpp>

extern "C" void cagraIndexDestroy(cagraIndex index)
{
  if (index.data_type.code == kDLFloat) {
    auto index_ptr = reinterpret_cast<raft::neighbors::cagra::index<float, uint32_t>*>(index.addr);
    delete index_ptr;
  } else if (index.data_type.code == kDLInt) {
    auto index_ptr = reinterpret_cast<raft::neighbors::cagra::index<int8_t, uint32_t>*>(index.addr);
    delete index_ptr;
  } else if (index.data_type.code == kDLUInt) {
    auto index_ptr =
      reinterpret_cast<raft::neighbors::cagra::index<uint8_t, uint32_t>*>(index.addr);
    delete index_ptr;
  }
}

template <typename T>
void* _build(raftResources_t res, cagraIndexParams params, DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;

  auto res_ptr                           = reinterpret_cast<raft::resources*>(res);
  auto index                             = new raft::neighbors::cagra::index<T, uint32_t>(*res_ptr);
  auto build_params                      = raft::neighbors::cagra::index_params();
  build_params.intermediate_graph_degree = params.intermediate_graph_degree;
  build_params.graph_degree              = params.graph_degree;
  build_params.build_algo =
    static_cast<raft::neighbors::cagra::graph_build_algo>(params.build_algo);
  build_params.nn_descent_niter = params.nn_descent_niter;

  if (dataset.device.device_type == kDLCUDA || dataset.device.device_type == kDLCUDAHost ||
      dataset.device.device_type == kDLCUDAManaged) {
    using mdspan_type = raft::device_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = raft::core::from_dlpack<mdspan_type>(dataset_tensor);
    raft::runtime::neighbors::cagra::build_device(*res_ptr, build_params, mds, *index);
  } else if (dataset.device.device_type == kDLCPU) {
    using mdspan_type = raft::host_matrix_view<T const, int64_t, raft::row_major>;
    auto mds          = raft::core::from_dlpack<mdspan_type>(dataset_tensor);
    raft::runtime::neighbors::cagra::build_host(*res_ptr, build_params, mds, *index);
  }

  return index;
}

extern "C" cagraIndex cagraBuild(raftResources_t res,
                                 cagraIndexParams params,
                                 DLManagedTensor* dataset_tensor)
{
  auto dataset = dataset_tensor->dl_tensor;
  RAFT_EXPECTS(dataset.dtype.lanes == 1, "More than 1 DL lanes not supported");

  cagraIndex index;
  if (dataset.dtype.code == kDLFloat && dataset.dtype.bits == 32) {
    index.addr           = reinterpret_cast<uintptr_t>(_build<float>(res, params, dataset_tensor));
    index.data_type.code = kDLFloat;
  } else if (dataset.dtype.code == kDLInt && dataset.dtype.bits == 8) {
    index.addr           = reinterpret_cast<uintptr_t>(_build<int8_t>(res, params, dataset_tensor));
    index.data_type.code = kDLInt;
  } else if (dataset.dtype.code == kDLUInt && dataset.dtype.bits == 8) {
    index.addr = reinterpret_cast<uintptr_t>(_build<uint8_t>(res, params, dataset_tensor));
    index.data_type.code = kDLUInt;
  } else {
    RAFT_FAIL("Unsupported DL dtype: %d and bits: %d", dataset.dtype.code, dataset.dtype.bits);
  }
  return index;
}
