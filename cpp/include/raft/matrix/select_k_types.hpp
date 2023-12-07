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

#pragma once

namespace raft::matrix {

/**
 * @defgroup select_k Batched-select k smallest or largest key/values
 * @{
 */

enum class SelectAlgo {
  kAuto,
  kRadix8bits,
  kRadix11bits,
  kRadix11bitsExtraPass,
  kWarpAuto,
  kWarpImmediate,
  kWarpFiltered,
  kWarpDistributed,
  kWarpDistributedShm,
};

inline auto operator<<(std::ostream& os, const SelectAlgo& algo) -> std::ostream&
{
  switch (algo) {
    case SelectAlgo::kAuto: return os << "kAuto";
    case SelectAlgo::kRadix8bits: return os << "kRadix8bits";
    case SelectAlgo::kRadix11bits: return os << "kRadix11bits";
    case SelectAlgo::kRadix11bitsExtraPass: return os << "kRadix11bitsExtraPass";
    case SelectAlgo::kWarpAuto: return os << "kWarpAuto";
    case SelectAlgo::kWarpImmediate: return os << "kWarpImmediate";
    case SelectAlgo::kWarpFiltered: return os << "kWarpFiltered";
    case SelectAlgo::kWarpDistributed: return os << "kWarpDistributed";
    case SelectAlgo::kWarpDistributedShm: return os << "kWarpDistributedShm";
    default: return os << "unknown enum value";
  }
}

/** @} */  // end of group select_k

}  // namespace raft::matrix
