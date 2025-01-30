/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

// We set RAFT_LOG_ACTIVE_LEVEL to a value that would enable testing trace and debug logs
// (otherwise trace and debug logs are desabled by default).
#undef RAFT_LOG_ACTIVE_LEVEL
#define RAFT_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_TRACE

#include <raft/core/logger.hpp>

#include <gtest/gtest.h>
#include <rapids_logger/logger.hpp>

#include <string>

namespace raft {

TEST(logger, Test)
{
  RAFT_LOG_CRITICAL("This is a critical message");
  RAFT_LOG_ERROR("This is an error message");
  RAFT_LOG_WARN("This is a warning message");
  RAFT_LOG_INFO("This is an info message");

  default_logger().set_level(rapids_logger::level_enum::warn);
  ASSERT_EQ(rapids_logger::level_enum::warn, default_logger().level());
  default_logger().set_level(rapids_logger::level_enum::info);
  ASSERT_EQ(rapids_logger::level_enum::info, default_logger().level());

  ASSERT_FALSE(default_logger().should_log(rapids_logger::level_enum::trace));
  ASSERT_FALSE(default_logger().should_log(rapids_logger::level_enum::debug));
  ASSERT_TRUE(default_logger().should_log(rapids_logger::level_enum::info));
  ASSERT_TRUE(default_logger().should_log(rapids_logger::level_enum::warn));
}

std::string logged = "";
void exampleCallback(int lvl, const char* msg) { logged = std::string(msg); }

int flushCount = 0;
void exampleFlush() { ++flushCount; }

class loggerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    flushCount = 0;
    logged     = "";
    default_logger().set_level(rapids_logger::level_enum::trace);
  }

  void TearDown() override
  {
    default_logger().sinks().pop_back();
    default_logger().set_level(rapids_logger::level_enum::info);
  }
};

// The logging macros depend on `RAFT_LOG_ACTIVE_LEVEL` as well as the logger verbosity;
// The verbosity is set to `RAFT_LOG_LEVEL_TRACE`, but `RAFT_LOG_ACTIVE_LEVEL` is set outside of
// here.
auto check_if_logged(const std::string& msg, rapids_logger::level_enum log_level_def) -> bool
{
  bool actually_logged  = logged.find(msg) != std::string::npos;
  bool should_be_logged = RAFT_LOG_ACTIVE_LEVEL <= static_cast<int>(log_level_def);
  return actually_logged == should_be_logged;
}

TEST_F(loggerTest, callback)
{
  std::string testMsg;
  default_logger().sinks().push_back(
    std::make_shared<rapids_logger::callback_sink_mt>(exampleCallback));

  testMsg = "This is a critical message";
  RAFT_LOG_CRITICAL(testMsg.c_str());
  ASSERT_TRUE(check_if_logged(testMsg, rapids_logger::level_enum::critical));

  testMsg = "This is an error message";
  RAFT_LOG_ERROR(testMsg.c_str());
  ASSERT_TRUE(check_if_logged(testMsg, rapids_logger::level_enum::error));

  testMsg = "This is a warning message";
  RAFT_LOG_WARN(testMsg.c_str());
  ASSERT_TRUE(check_if_logged(testMsg, rapids_logger::level_enum::warn));

  testMsg = "This is an info message";
  RAFT_LOG_INFO(testMsg.c_str());
  ASSERT_TRUE(check_if_logged(testMsg, rapids_logger::level_enum::info));

  testMsg = "This is a debug message";
  RAFT_LOG_DEBUG(testMsg.c_str());
  ASSERT_TRUE(check_if_logged(testMsg, rapids_logger::level_enum::debug));

  testMsg = "This is a trace message";
  RAFT_LOG_TRACE(testMsg.c_str());
  ASSERT_TRUE(check_if_logged(testMsg, rapids_logger::level_enum::trace));
}

TEST_F(loggerTest, flush)
{
  default_logger().sinks().push_back(
    std::make_shared<rapids_logger::callback_sink_mt>(exampleCallback, exampleFlush));
  default_logger().flush();
  ASSERT_EQ(1, flushCount);
}

}  // namespace raft
