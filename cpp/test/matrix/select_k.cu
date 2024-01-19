/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

auto inputs_random_longlist = testing::Values(select::params{1, 130, 15, false},
                                              select::params{1, 128, 15, false},
                                              select::params{20, 700, 1, true},
                                              select::params{20, 700, 2, true},
                                              select::params{20, 700, 3, true},
                                              select::params{20, 700, 4, true},
                                              select::params{20, 700, 5, true},
                                              select::params{20, 700, 6, true},
                                              select::params{20, 700, 7, true},
                                              select::params{20, 700, 8, true},
                                              select::params{20, 700, 9, true},
                                              select::params{20, 700, 10, true, false},
                                              select::params{20, 700, 11, true},
                                              select::params{20, 700, 12, true},
                                              select::params{20, 700, 16, true},
                                              select::params{100, 1700, 17, true},
                                              select::params{100, 1700, 31, true, false},
                                              select::params{100, 1700, 32, false},
                                              select::params{100, 1700, 33, false},
                                              select::params{100, 1700, 63, false},
                                              select::params{100, 1700, 64, false, false},
                                              select::params{100, 1700, 65, false},
                                              select::params{100, 1700, 255, true},
                                              select::params{100, 1700, 256, true},
                                              select::params{100, 1700, 511, false},
                                              select::params{100, 1700, 512, true},
                                              select::params{100, 1700, 1023, false, false},
                                              select::params{100, 1700, 1024, true},
                                              select::params{100, 1700, 1700, true});

auto inputs_random_largesize = testing::Values(select::params{100, 100000, 1, true},
                                               select::params{100, 100000, 2, true},
                                               select::params{100, 100000, 3, true, false},
                                               select::params{100, 100000, 7, true},
                                               select::params{100, 100000, 16, true},
                                               select::params{100, 100000, 31, true},
                                               select::params{100, 100000, 32, true, false},
                                               select::params{100, 100000, 60, true},
                                               select::params{100, 100000, 100, true, false},
                                               select::params{100, 100000, 200, true},
                                               select::params{100000, 100, 100, false},
                                               select::params{1, 1000000000, 1, true},
                                               select::params{1, 1000000000, 16, false, false},
                                               select::params{1, 1000000000, 64, false},
                                               select::params{1, 1000000000, 128, true, false},
                                               select::params{1, 1000000000, 256, false, false});

auto inputs_random_largek = testing::Values(select::params{100, 100000, 1000, true},
                                            select::params{100, 100000, 2000, false},
                                            select::params{100, 100000, 100000, true, false},
                                            select::params{100, 100000, 2048, false},
                                            select::params{100, 100000, 1237, true});

auto inputs_random_many_infs =
  testing::Values(select::params{10, 100000, 1, true, false, false, true, 0.9},
                  select::params{10, 100000, 16, true, false, false, true, 0.9},
                  select::params{10, 100000, 64, true, false, false, true, 0.9},
                  select::params{10, 100000, 128, true, false, false, true, 0.9},
                  select::params{10, 100000, 256, true, false, false, true, 0.9},
                  select::params{1000, 10000, 1, true, false, false, true, 0.9},
                  select::params{1000, 10000, 16, true, false, false, true, 0.9},
                  select::params{1000, 10000, 64, true, false, false, true, 0.9},
                  select::params{1000, 10000, 128, true, false, false, true, 0.9},
                  select::params{1000, 10000, 256, true, false, false, true, 0.9},
                  select::params{10, 100000, 1, true, false, false, true, 0.999},
                  select::params{10, 100000, 16, true, false, false, true, 0.999},
                  select::params{10, 100000, 64, true, false, false, true, 0.999},
                  select::params{10, 100000, 128, true, false, false, true, 0.999},
                  select::params{10, 100000, 256, true, false, false, true, 0.999},
                  select::params{1000, 10000, 1, true, false, false, true, 0.999},
                  select::params{1000, 10000, 16, true, false, false, true, 0.999},
                  select::params{1000, 10000, 64, true, false, false, true, 0.999},
                  select::params{1000, 10000, 128, true, false, false, true, 0.999},
                  select::params{1000, 10000, 256, true, false, false, true, 0.999});

using ReferencedRandomFloatInt =
  SelectK<float, uint32_t, with_ref<SelectAlgo::kAuto>::params_random>;
TEST_P(ReferencedRandomFloatInt, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                          // NOLINT
  SelectK,
  ReferencedRandomFloatInt,
  testing::Combine(inputs_random_longlist,
                   testing::Values(SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass,
                                   SelectAlgo::kWarpImmediate,
                                   SelectAlgo::kWarpFiltered,
                                   SelectAlgo::kWarpDistributed,
                                   SelectAlgo::kWarpDistributedShm)));

using ReferencedRandomDoubleSizeT =
  SelectK<double, int64_t, with_ref<SelectAlgo::kAuto>::params_random>;
TEST_P(ReferencedRandomDoubleSizeT, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                             // NOLINT
  SelectK,
  ReferencedRandomDoubleSizeT,
  testing::Combine(inputs_random_longlist,
                   testing::Values(SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass,
                                   SelectAlgo::kWarpImmediate,
                                   SelectAlgo::kWarpFiltered,
                                   SelectAlgo::kWarpDistributed,
                                   SelectAlgo::kWarpDistributedShm)));

using ReferencedRandomDoubleInt =
  SelectK<double, uint32_t, with_ref<SelectAlgo::kRadix11bits>::params_random>;
TEST_P(ReferencedRandomDoubleInt, LargeSize) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                                 // NOLINT
  SelectK,
  ReferencedRandomDoubleInt,
  testing::Combine(inputs_random_largesize,
                   testing::Values(SelectAlgo::kWarpAuto,
                                   SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass)));

using ReferencedRandomFloatIntkWarpsortAsGT =
  SelectK<float, uint32_t, with_ref<SelectAlgo::kWarpImmediate>::params_random>;
TEST_P(ReferencedRandomFloatIntkWarpsortAsGT, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                                       // NOLINT
  SelectK,
  ReferencedRandomFloatIntkWarpsortAsGT,
  testing::Combine(inputs_random_many_infs,
                   testing::Values(SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass)));

}  // namespace raft::matrix
