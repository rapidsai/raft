/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
  T x{args...};
  auto s = serialize(x);
  auto y = deserialize<T>(s);
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
struct serial<not_initializable> {
  static auto to_bytes(uint8_t* out, const not_initializable& obj) -> size_t
  {
    if (out) { *reinterpret_cast<uint32_t*>(out) = obj.content; }
    return sizeof(not_initializable);
  }

  static auto from_bytes(not_initializable* obj, const uint8_t* in) -> size_t
  {
    auto val = *reinterpret_cast<const uint32_t*>(in);
    if (val == 7) { throw raft::exception("I don't like this number at all"); }
    new (obj) not_initializable{val};
    return sizeof(not_initializable);
  }
};

}  // namespace detail

// Here, with the help of `not_initializable` we test two things:
//  1. No extra unwanted constructors are called within `deserialize`.
//  2. `deserialize` returns by value without extra copy constructors (copy elision)
TEST(serialize, not_initializable)
{
  ASSERT_EQ(not_initializable::count, 0);
  {
    not_initializable x{17};
    ASSERT_EQ(not_initializable::count, 1);
    auto s = serialize(x);
    ASSERT_EQ(not_initializable::count, 1);
    auto y = deserialize<not_initializable>(s);
    ASSERT_EQ(not_initializable::count, 2);
    ASSERT_EQ(x.content, y.content);
  }
  ASSERT_EQ(not_initializable::count, 0);
  {
    not_initializable x{7};
    ASSERT_EQ(not_initializable::count, 1);
    auto s = serialize(x);
    ASSERT_EQ(not_initializable::count, 1);
    try {
      auto y = deserialize<not_initializable>(s);
      // shouldn't be reachable anyway, because it doesn't like number 7.
      ASSERT_EQ(not_initializable::count, 2);
      ASSERT_EQ(x.content, y.content);
    } catch (raft::exception&) {
      ASSERT_EQ(not_initializable::count, 1);
    }
    ASSERT_EQ(not_initializable::count, 1);
  }
  ASSERT_EQ(not_initializable::count, 0);
}

TEST(serialize, mdarray)
{
  handle_t handle;
  auto exts = make_extents<int>(2, 3, 4);
  auto x    = make_device_mdarray<double>(handle, exts);
  auto s    = serialize(x, handle);
  auto y    = deserialize<decltype(x)>(s, handle);
  ASSERT_EQ(x.extents(), y.extents());
  for (int i = 0; i < exts.extent(2); i++) {
    ASSERT_EQ(x(0, 2, i), y(0, 2, i));
  }
}

TEST(serialize, device_uvector)
{
  auto stream = rmm::cuda_stream_default;
  auto x      = rmm::device_uvector<int16_t>(10, stream);
  auto s      = serialize(x, stream);
  auto y      = deserialize<decltype(x)>(s, stream);
  ASSERT_EQ(x.size(), y.size());
}

}  // namespace raft
