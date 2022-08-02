/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/serialization.hpp>

#include <gtest/gtest.h>

namespace raft {

template <typename T, typename... Args>
void test_serialization(Args... args)
{
  handle_t handle;
  T x{args...};
  auto s = serialize(handle, x);
  auto y = deserialize<T>(handle, s);
  ASSERT_EQ(x, y);
}

#define ARGS_HEAD(a, ...) a
#define ARGS_TAIL(a, ...) __VA_ARGS__
#define TEST_SIMPLE(...)                                                \
  TEST(serialize, ARGS_HEAD(__VA_ARGS__))                               \
  {                                                                     \
    test_serialization<ARGS_HEAD(__VA_ARGS__)>(ARGS_TAIL(__VA_ARGS__)); \
  }  // NOLINT

TEST_SIMPLE(int, 42);
TEST_SIMPLE(uint64_t, 1u);
TEST_SIMPLE(float, 3.0f);

struct not_initializable {
  const uint32_t content;
  static inline int count = 0;  // NOLINT

  explicit not_initializable(uint32_t c) : content(c) { count++; }
  not_initializable()                         = delete;
  not_initializable(const not_initializable&) = delete;
  not_initializable(not_initializable&& other) : content(other.content) { count++; }
  auto operator=(const not_initializable&) -> not_initializable& = delete;
  auto operator=(not_initializable&&) -> not_initializable& = delete;
  ~not_initializable() { count--; }
};

namespace detail {

template <>
struct serialize<not_initializable> {
  static auto run(const handle_t& handle, const not_initializable& obj, uint8_t* out) -> size_t
  {
    if (out) { *reinterpret_cast<uint32_t*>(out) = obj.content; }
    return sizeof(not_initializable);
  }
};

template <>
struct deserialize<not_initializable> {
  static void run(const handle_t& handle, not_initializable* obj, const uint8_t* in)
  {
    new (obj) not_initializable{*reinterpret_cast<const uint32_t*>(in)};
  }
};

}  // namespace detail

// Here, with the help of `not_initializable` we test two things:
//  1. No extra unwanted constructors are called within `deserialize`.
//  2. `deserialize` returns by value without extra copy constructors (copy elision)
TEST(serialize, not_initializable)
{
  handle_t handle;
  ASSERT_EQ(not_initializable::count, 0);
  {
    not_initializable x{17};
    ASSERT_EQ(not_initializable::count, 1);
    auto s = serialize(handle, x);
    ASSERT_EQ(not_initializable::count, 1);
    auto y = deserialize<not_initializable>(handle, s);
    ASSERT_EQ(not_initializable::count, 2);
    ASSERT_EQ(x.content, y.content);
  }
  ASSERT_EQ(not_initializable::count, 0);
}

TEST(serialize, mdarray)
{
  handle_t handle;
  auto exts = make_extents<int>(2, 3, 4);
  auto x    = make_device_mdarray<double>(handle, exts);
  auto s    = serialize(handle, x);
  auto y    = deserialize<decltype(x)>(handle, s);
  ASSERT_EQ(x.extents(), y.extents());
  for (int i = 0; i < exts.extent(2); i++) {
    ASSERT_EQ(x(0, 2, i), y(0, 2, i));
  }
}

}  // namespace raft
