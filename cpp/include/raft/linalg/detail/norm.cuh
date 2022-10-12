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

#pragma once

#include <raft/linalg/reduce.cuh>

namespace raft {
namespace linalg {
namespace detail {

/** different types of norms supported on the input buffers */
enum NormType { L1Norm = 0, L2Norm };

template <typename Type, typename IdxType, typename Lambda>
void rowNormCaller(Type* dots,
                   const Type* data,
                   IdxType D,
                   IdxType N,
                   NormType type,
                   bool rowMajor,
                   cudaStream_t stream,
                   Lambda fin_op)
{
  switch (type) {
    case L1Norm:
      raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                data,
                                                D,
                                                N,
                                                (Type)0,
                                                rowMajor,
                                                true,
                                                stream,
                                                false,
                                                raft::L1Op<Type, IdxType>(),
                                                raft::Sum<Type>(),
                                                fin_op);
      break;
    case L2Norm:
      raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                data,
                                                D,
                                                N,
                                                (Type)0,
                                                rowMajor,
                                                true,
                                                stream,
                                                false,
                                                raft::L2Op<Type>(),
                                                raft::Sum<Type>(),
                                                fin_op);
      break;
    default: ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

template <typename Type, typename IdxType, typename Lambda>
void colNormCaller(Type* dots,
                   const Type* data,
                   IdxType D,
                   IdxType N,
                   NormType type,
                   bool rowMajor,
                   cudaStream_t stream,
                   Lambda fin_op)
{
  switch (type) {
    case L1Norm:
      raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                data,
                                                D,
                                                N,
                                                (Type)0,
                                                rowMajor,
                                                false,
                                                stream,
                                                false,
                                                raft::L1Op<Type, IdxType>(),
                                                raft::Sum<Type>(),
                                                fin_op);
      break;
    case L2Norm:
      raft::linalg::reduce<Type, Type, IdxType>(dots,
                                                data,
                                                D,
                                                N,
                                                (Type)0,
                                                rowMajor,
                                                false,
                                                stream,
                                                false,
                                                raft::L2Op<Type, IdxType>(),
                                                raft::Sum<Type>(),
                                                fin_op);
      break;
    default: ASSERT(false, "Invalid norm type passed! [%d]", type);
  };
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
