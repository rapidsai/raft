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

#include <gtest/gtest.h>
#include <raft/common/logger.hpp>
#include <string>

namespace raft {

TEST(logger, Test)
{
  RAFT_LOG_CRITICAL("This is a critical message");
  RAFT_LOG_ERROR("This is an error message");
  RAFT_LOG_WARN("This is a warning message");
  RAFT_LOG_INFO("This is an info message");

  logger::get(RAFT_NAME).set_level(RAFT_LEVEL_WARN);
  ASSERT_EQ(RAFT_LEVEL_WARN, logger::get(RAFT_NAME).get_level());
  logger::get(RAFT_NAME).set_level(RAFT_LEVEL_INFO);
  ASSERT_EQ(RAFT_LEVEL_INFO, logger::get(RAFT_NAME).get_level());

  ASSERT_FALSE(logger::get(RAFT_NAME).should_log_for(RAFT_LEVEL_TRACE));
  ASSERT_FALSE(logger::get(RAFT_NAME).should_log_for(RAFT_LEVEL_DEBUG));
  ASSERT_TRUE(logger::get(RAFT_NAME).should_log_for(RAFT_LEVEL_INFO));
  ASSERT_TRUE(logger::get(RAFT_NAME).should_log_for(RAFT_LEVEL_WARN));
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
    logger::get(RAFT_NAME).set_level(RAFT_LEVEL_TRACE);
  }

  void TearDown() override
  {
    logger::get(RAFT_NAME).set_callback(nullptr);
    logger::get(RAFT_NAME).set_flush(nullptr);
    logger::get(RAFT_NAME).set_level(RAFT_LEVEL_INFO);
  }
};

TEST_F(loggerTest, callback)
{
  std::string testMsg;
  logger::get(RAFT_NAME).set_callback(exampleCallback);

  testMsg = "This is a critical message";
  RAFT_LOG_CRITICAL(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is an error message";
  RAFT_LOG_ERROR(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is a warning message";
  RAFT_LOG_WARN(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is an info message";
  RAFT_LOG_INFO(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);

  testMsg = "This is a debug message";
  RAFT_LOG_DEBUG(testMsg.c_str());
  ASSERT_TRUE(logged.find(testMsg) != std::string::npos);
}

TEST_F(loggerTest, flush)
{
  logger::get(RAFT_NAME).set_flush(exampleFlush);
  logger::get(RAFT_NAME).flush();
  ASSERT_EQ(1, flushCount);
}

}  // namespace raft