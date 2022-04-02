/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

namespace raft {
namespace random {
namespace detail {

/** all different generator types used */
enum GeneratorType {
  /** curand-based philox generator */
  GenPhilox = 0,
  /** Permuted Congruential Generator */
  GenPC
};

struct RngState {
  uint64_t seed;
  uint64_t base_subsequence;
  GeneratorType type;

  void advance(uint64_t max_uniq_subsequences_used,
               uint64_t max_numbers_generated_per_subsequence = 0)
  {
    base_subsequence += max_uniq_subsequences_used;
  }
};

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
