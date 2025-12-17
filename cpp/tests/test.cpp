/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft.hpp>

#include <gtest/gtest.h>

#include <iostream>

namespace raft {

TEST(Raft, print) { std::cout << test_raft() << std::endl; }

}  // namespace raft
