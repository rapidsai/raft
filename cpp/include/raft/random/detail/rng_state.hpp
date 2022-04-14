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
  explicit RngState(uint64_t _seed) : seed(_seed) {}
  RngState(uint64_t _seed, GeneratorType _type) : seed(_seed), type(_type) {}
  RngState(uint64_t _seed, uint64_t _base_subsequence, GeneratorType _type) : seed(_seed), base_subsequence(_base_subsequence), type(_type) {}

  uint64_t seed{0};
  uint64_t base_subsequence{0};
  GeneratorType type{GeneratorType::GenPC};

  void advance(uint64_t max_uniq_subsequences_used,
               uint64_t max_numbers_generated_per_subsequence = 0)
  {
    base_subsequence += max_uniq_subsequences_used;
  }

  template <typename IdxT>
  void affine_transform_params(IdxT n, IdxT& a, IdxT& b)
  {
    // always keep 'a' to be coprime to 'n'
    std::mt19937_64 mt_rng(seed + base_subsequence);
    a = mt_rng() % n;
    while (gcd(a, n) != 1) {
      ++a;
      if (a >= n) a = 0;
    }
    // the bias term 'b' can be any number in the range of [0, n)
    b = mt_rng() % n;
  }
};

};  // end namespace detail
};  // end namespace random
};  // end namespace raft
