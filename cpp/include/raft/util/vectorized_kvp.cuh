/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/kvp.hpp>

#include <type_traits>

namespace raft {
/**
 * Generic IOType specializations for ALL KeyValuePair<K, V> types based on sizeof.
 *
 * 4-byte KVP (e.g., <int16_t,int16_t>):
 *   - VecLen=1: int32_t (4 bytes, load 1 KVP)
 *   - VecLen=2: int2 (8 bytes, load 2 KVPs)
 *   - VecLen=4: int4 (16 bytes, load 4 KVPs)
 *
 * 8-byte KVP (e.g., <int,float>, <int,int>, <uint32_t,float>):
 *   - VecLen=1: int2 (8 bytes, load 1 KVP)
 *   - VecLen=2: int4 (16 bytes, load 2 KVPs)
 *
 * 16-byte KVP (e.g., <int64_t, double>, <int, double>):
 *   - VecLen=1: int4 (16 bytes, load 1 KVP)
 */

// 4-byte KVP specializations
template <typename K, typename V>
requires(sizeof(KeyValuePair<K, V>) == 4) struct IOType<KeyValuePair<K, V>, 1> {
  static_assert(std::is_trivially_copyable_v<KeyValuePair<K, V>>);
  using Type = int32_t;
};

template <typename K, typename V>
requires(sizeof(KeyValuePair<K, V>) == 4) struct IOType<KeyValuePair<K, V>, 2> {
  static_assert(std::is_trivially_copyable_v<KeyValuePair<K, V>>);
  using Type = int2;
};

template <typename K, typename V>
requires(sizeof(KeyValuePair<K, V>) == 4) struct IOType<KeyValuePair<K, V>, 4> {
  static_assert(std::is_trivially_copyable_v<KeyValuePair<K, V>>);
  using Type = int4;
};

// 8-byte KVP specializations
template <typename K, typename V>
requires(sizeof(KeyValuePair<K, V>) == 8) struct IOType<KeyValuePair<K, V>, 1> {
  static_assert(std::is_trivially_copyable_v<KeyValuePair<K, V>>);
  using Type = int2;
};

template <typename K, typename V>
requires(sizeof(KeyValuePair<K, V>) == 8) struct IOType<KeyValuePair<K, V>, 2> {
  static_assert(std::is_trivially_copyable_v<KeyValuePair<K, V>>);
  using Type = int4;
};

// 16-byte KVP specializations
template <typename K, typename V>
requires(sizeof(KeyValuePair<K, V>) == 16) struct IOType<KeyValuePair<K, V>, 1> {
  static_assert(std::is_trivially_copyable_v<KeyValuePair<K, V>>);
  using Type = int4;
};

}  // namespace raft
