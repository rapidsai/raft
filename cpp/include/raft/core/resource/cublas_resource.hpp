/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "resource.hpp"
#include <cublas_v2.h>

class cublas_resource_t : public resource_t {
 public:
  typename res_t = cublasHandle_t

  cublas_resource_t(cudaStream_t stream)
  {
  }

  res_t* get_resource() { return &cublas_handle; }

 private:
  res_t cublas_handle;
};