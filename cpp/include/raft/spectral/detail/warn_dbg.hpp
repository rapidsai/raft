/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <stdexcept>
#include <string>

#ifdef DEBUG
#define COUT() (std::cout)
#define CERR() (std::cerr)

// nope:
//
#define WARNING(message)                                                  \
  do {                                                                    \
    std::stringstream ss;                                                 \
    ss << "Warning (" << __FILE__ << ":" << __LINE__ << "): " << message; \
    CERR() << ss.str() << std::endl;                                      \
  } while (0)
#else  // DEBUG
#define WARNING(message)
#endif
