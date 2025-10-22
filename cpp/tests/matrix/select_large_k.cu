/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "select_k.cuh"

namespace raft::matrix {

auto inputs_random_largek = testing::Values(select::params{100, 100000, 1000, true},
                                            select::params{100, 100000, 2000, false},
                                            select::params{100, 100000, 100000, true, false},
                                            select::params{100, 100000, 2048, false},
                                            select::params{100, 100000, 1237, true});

using ReferencedRandomFloatSizeT =
  SelectK<float, int64_t, with_ref<SelectAlgo::kRadix8bits>::params_random>;
TEST_P(ReferencedRandomFloatSizeT, LargeK) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(SelectK,                       // NOLINT
                        ReferencedRandomFloatSizeT,
                        testing::Combine(inputs_random_largek,
                                         testing::Values(SelectAlgo::kRadix11bits,
                                                         SelectAlgo::kRadix11bitsExtraPass)));

}  // namespace raft::matrix
