/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __RNG_STATE_H
#define __RNG_STATE_H

#pragma once

#include <cstdint>

namespace raft {
namespace random {

/** all different generator types used */
enum GeneratorType {
  /** curand-based philox generator */
  GenPhilox = 0,
  /** Permuted Congruential Generator */
  GenPC
};

/**
 * The RNG state used to keep RNG state around on the host.
 */
struct RngState {
  explicit RngState(uint64_t _seed) : seed(_seed) {}
  RngState(uint64_t _seed, GeneratorType _type) : seed(_seed), type(_type) {}
  RngState(uint64_t _seed, uint64_t _base_subsequence, GeneratorType _type)
    : seed(_seed), base_subsequence(_base_subsequence), type(_type)
  {
  }

  uint64_t seed{0};
  uint64_t base_subsequence{0};
  /**
   * The generator type. PCGenerator has been extensively tested and is faster
   * than Philox, thus we use it as the default.
   */
  GeneratorType type{GeneratorType::GenPC};

  void advance(uint64_t max_uniq_subsequences_used,
               uint64_t max_numbers_generated_per_subsequence = 0)
  {
    base_subsequence += max_uniq_subsequences_used;
  }
};

};  // end namespace random
};  // end namespace raft

#endif
