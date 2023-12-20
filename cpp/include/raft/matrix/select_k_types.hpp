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
#include <type_traits>

namespace raft::matrix {

/**
 * @defgroup select_k Batched-select k smallest or largest key/values
 * @{
 */

enum class SelectAlgo : uint8_t {
  kAuto                 = 0,
  kRadix8bits           = 1,
  kRadix11bits          = 2,
  kRadix11bitsExtraPass = 3,
  kWarpAuto             = 4,
  kWarpImmediate        = 5,
  kWarpFiltered         = 6,
  kWarpDistributed      = 7,
  kWarpDistributedShm   = 8,
};

inline auto operator<<(std::ostream& os, const SelectAlgo& algo) -> std::ostream&
{
  auto underlying_value = static_cast<std::underlying_type<SelectAlgo>::type>(algo);

  switch (algo) {
    case SelectAlgo::kAuto: return os << "kAuto=" << underlying_value;
    case SelectAlgo::kRadix8bits: return os << "kRadix8bits=" << underlying_value;
    case SelectAlgo::kRadix11bits: return os << "kRadix11bits=" << underlying_value;
    case SelectAlgo::kRadix11bitsExtraPass:
      return os << "kRadix11bitsExtraPass=" << underlying_value;
    case SelectAlgo::kWarpAuto: return os << "kWarpAuto=" << underlying_value;
    case SelectAlgo::kWarpImmediate: return os << "kWarpImmediate=" << underlying_value;
    case SelectAlgo::kWarpFiltered: return os << "kWarpFiltered=" << underlying_value;
    case SelectAlgo::kWarpDistributed: return os << "kWarpDistributed=" << underlying_value;
    case SelectAlgo::kWarpDistributedShm: return os << "kWarpDistributedShm=" << underlying_value;
    default: throw std::invalid_argument("invalid value for SelectAlgo");
  }
}

/** @} */  // end of group select_k

}  // namespace raft::matrix
