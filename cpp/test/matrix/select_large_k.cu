/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "select_k.cuh"

namespace raft::matrix {

auto inputs_random_largek = testing::Values(select::params{100, 100000, 1000, true},
                                            select::params{100, 100000, 2000, false},
                                            select::params{100, 100000, 100000, true, false},
                                            select::params{100, 100000, 2048, false},
                                            select::params{100, 100000, 1237, true});

using ReferencedRandomFloatSizeT =
  SelectK<float, int64_t, with_ref<select::Algo::kRadix8bits>::params_random>;
TEST_P(ReferencedRandomFloatSizeT, LargeK) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(SelectK,                       // NOLINT
                        ReferencedRandomFloatSizeT,
                        testing::Combine(inputs_random_largek,
                                         testing::Values(select::Algo::kRadix11bits,
                                                         select::Algo::kRadix11bitsExtraPass)));

}  // namespace raft::matrix
