/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#pragma once

#include <rmm/cuda_stream_view.hpp>

namespace raft {
namespace common {
namespace detail {

#ifdef NVTX_ENABLED

#include <nvToolsExt.h>
#include <stdint.h>
#include <stdlib.h>
#include <mutex>
#include <string>
#include <unordered_map>

/**
 * @brief An internal struct to store associated state with the color
 * generator
 */
struct ColorGenState {
  /** collection of all tagged colors generated so far */
  static inline std::unordered_map<std::string, uint32_t> allColors;
  /** mutex for accessing the above map */
  static inline std::mutex mapMutex;
  /** saturation */
  static inline constexpr float S = 0.9f;
  /** value */
  static inline constexpr float V = 0.85f;
  /** golden ratio */
  static inline constexpr float Phi = 1.61803f;
  /** inverse golden ratio */
  static inline constexpr float InvPhi = 1.f / Phi;
};

// all h, s, v are in range [0, 1]
// Ref: http://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB
inline uint32_t hsv2rgb(float h, float s, float v)
{
  uint32_t out = 0xff000000u;
  if (s <= 0.0f) { return out; }
  // convert hue from [0, 1] range to [0, 360]
  float h_deg = h * 360.f;
  if (0.f > h_deg || h_deg >= 360.f) h_deg = 0.f;
  h_deg /= 60.f;
  int h_range = (int)h_deg;
  float h_mod = h_deg - h_range;
  float x     = v * (1.f - s);
  float y     = v * (1.f - (s * h_mod));
  float z     = v * (1.f - (s * (1.f - h_mod)));
  float r, g, b;
  switch (h_range) {
    case 0:
      r = v;
      g = z;
      b = x;
      break;
    case 1:
      r = y;
      g = v;
      b = x;
      break;
    case 2:
      r = x;
      g = v;
      b = z;
      break;
    case 3:
      r = x;
      g = y;
      b = v;
      break;
    case 4:
      r = z;
      g = x;
      b = v;
      break;
    case 5:
    default:
      r = v;
      g = x;
      b = y;
      break;
  }
  out |= (uint32_t(r * 256.f) << 16);
  out |= (uint32_t(g * 256.f) << 8);
  out |= uint32_t(b * 256.f);
  return out;
}

/**
 * @brief Helper method to generate 'visually distinct' colors.
 * Inspired from https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
 * However, if an associated tag is passed, it will look up in its history for
 * any generated color against this tag and if found, just returns it, else
 * generates a new color, assigns a tag to it and stores it for future usage.
 * Such a thing is very useful for nvtx markers where the ranges associated
 * with a specific tag should ideally get the same color for the purpose of
 * visualizing it on nsight-systems timeline.
 * @param tag look for any previously generated colors with this tag or
 * associate the currently generated color with it
 * @return returns 32b RGB integer with alpha channel set of 0xff
 */
inline uint32_t generateNextColor(const std::string& tag)
{
  // std::unordered_map<std::string, uint32_t> ColorGenState::allColors;
  // std::mutex ColorGenState::mapMutex;

  std::lock_guard<std::mutex> guard(ColorGenState::mapMutex);
  if (!tag.empty()) {
    auto itr = ColorGenState::allColors.find(tag);
    if (itr != ColorGenState::allColors.end()) { return itr->second; }
  }
  float h = rand() * 1.f / RAND_MAX;
  h += ColorGenState::InvPhi;
  if (h >= 1.f) h -= 1.f;
  auto rgb = hsv2rgb(h, ColorGenState::S, ColorGenState::V);
  if (!tag.empty()) { ColorGenState::allColors[tag] = rgb; }
  return rgb;
}

static inline nvtxDomainHandle_t domain = nvtxDomainCreateA("application");

inline void pushRange_name(const char* name)
{
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version               = NVTX_VERSION;
  eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType             = NVTX_COLOR_ARGB;
  eventAttrib.color                 = generateNextColor(name);
  eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii         = name;
  nvtxDomainRangePushEx(domain, &eventAttrib);
}

template <typename... Args>
inline void pushRange(const char* format, Args... args)
{
  if constexpr (sizeof...(args) > 0) {
    int length = std::snprintf(nullptr, 0, format, args...);
    assert(length >= 0);
    auto buf = std::make_unique<char[]>(length + 1);
    std::snprintf(buf.get(), length + 1, format, args...);
    pushRange_name(buf.get());
  } else {
    pushRange_name(format);
  }
}

template <typename... Args>
inline void pushRange(rmm::cuda_stream_view stream, const char* format, Args... args)
{
  stream.synchronize();
  pushRange(format, args...);
}

inline void popRange() { nvtxDomainRangePop(domain); }

inline void popRange(rmm::cuda_stream_view stream)
{
  stream.synchronize();
  popRange();
}

#else  // NVTX_ENABLED

template <typename... Args>
inline void pushRange(const char* format, Args... args)
{
}

template <typename... Args>
inline void pushRange(rmm::cuda_stream_view stream, const char* format, Args... args)
{
}

inline void popRange() {}

inline void popRange(rmm::cuda_stream_view stream) {}

#endif  // NVTX_ENABLED

}  // namespace detail
}  // namespace common
}  // namespace raft
