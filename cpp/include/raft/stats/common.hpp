/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/util/cudart_utils.hpp>

// This file is a shameless amalgamation of independent works done by
// Lars Nyland and Andy Adinets

///@todo: add cub's histogram as another option

namespace raft {
namespace stats {

/** Default mapper which just returns the value of the data itself */
template <typename DataT, typename IdxT>
struct IdentityBinner {
  DI int operator()(DataT val, IdxT row, IdxT col) { return int(val); }
};

/** Types of support histogram implementations */
enum HistType {
  /** shared mem atomics but with bins to be 1b int's */
  HistTypeSmemBits1 = 1,
  /** shared mem atomics but with bins to be 2b int's */
  HistTypeSmemBits2 = 2,
  /** shared mem atomics but with bins to be 4b int's */
  HistTypeSmemBits4 = 4,
  /** shared mem atomics but with bins to ba 1B int's */
  HistTypeSmemBits8 = 8,
  /** shared mem atomics but with bins to be 2B int's */
  HistTypeSmemBits16 = 16,
  /** use only global atomics */
  HistTypeGmem,
  /** uses shared mem atomics to reduce global traffic */
  HistTypeSmem,
  /**
   * uses shared mem atomics with match_any intrinsic to further reduce shared
   * memory traffic. This can only be enabled on Volta and later architectures.
   * If one tries to enable this for older arch's, it will fall back to
   * `HistTypeSmem`.
   * @note This is to be used only when the input dataset leads to a lot of
   *       repetitions in a given warp, else, this algo can be much slower than
   *       `HistTypeSmem`!
   */
  HistTypeSmemMatchAny,
  /** builds a hashmap of active bins in shared mem */
  HistTypeSmemHash,
  /** decide at runtime the best algo for the given inputs */
  HistTypeAuto
};

/// Supported types of information criteria
enum IC_Type { AIC, AICc, BIC };

};  // end namespace stats
};  // end namespace raft
