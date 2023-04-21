/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

namespace raft::common::nvtx::detail {

#ifdef NVTX_ENABLED

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <type_traits>
#include <unordered_map>

/**
 * @brief An internal struct to store associated state with the color
 * generator
 */
struct color_gen_state {
  /** collection of all tagged colors generated so far */
  static inline std::unordered_map<std::string, uint32_t> all_colors_;
  /** mutex for accessing the above map */
  static inline std::mutex map_mutex_;
  /** saturation */
  static inline constexpr float kS = 0.9f;
  /** value */
  static inline constexpr float kV = 0.85f;
  /** golden ratio */
  static inline constexpr float kPhi = 1.61803f;
  /** inverse golden ratio */
  static inline constexpr float kInvPhi = 1.f / kPhi;
};

// all h, s, v are in range [0, 1]
// Ref: http://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB
inline auto hsv2rgb(float h, float s, float v) -> uint32_t
{
  uint32_t out = 0xff000000u;
  if (s <= 0.0f) { return out; }
  // convert hue from [0, 1] range to [0, 360]
  float h_deg = h * 360.f;
  if (0.f > h_deg || h_deg >= 360.f) h_deg = 0.f;
  h_deg /= 60.f;
  int h_range = static_cast<int>(h_deg);
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
inline auto generate_next_color(const std::string& tag) -> uint32_t
{
  // std::unordered_map<std::string, uint32_t> color_gen_state::all_colors_;
  // std::mutex color_gen_state::map_mutex_;

  std::lock_guard<std::mutex> guard(color_gen_state::map_mutex_);
  if (!tag.empty()) {
    auto itr = color_gen_state::all_colors_.find(tag);
    if (itr != color_gen_state::all_colors_.end()) { return itr->second; }
  }
  auto h = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  h += color_gen_state::kInvPhi;
  if (h >= 1.f) h -= 1.f;
  auto rgb = hsv2rgb(h, color_gen_state::kS, color_gen_state::kV);
  if (!tag.empty()) { color_gen_state::all_colors_[tag] = rgb; }
  return rgb;
}

template <typename Domain, typename = Domain>
struct domain_store {
  /* If `Domain::name` does not exist, this default instance is used and throws the error. */
  static_assert(sizeof(Domain) != sizeof(Domain),
                "Type used to identify a domain must contain a static member 'char const* name'");
  static inline auto value() -> const nvtxDomainHandle_t { return nullptr; }
};

template <typename Domain>
struct domain_store<
  Domain,
  /* Check if there exists `Domain::name` */
  std::enable_if_t<
    std::is_same<char const*, typename std::decay<decltype(Domain::name)>::type>::value,
    Domain>> {
  static inline auto value() -> const nvtxDomainHandle_t
  {
    // NB: static modifier ensures the domain is created only once
    static const nvtxDomainHandle_t kValue = nvtxDomainCreateA(Domain::name);
    return kValue;
  }
};

template <typename Domain>
inline void push_range_name(const char* name)
{
  nvtxEventAttributes_t event_attrib = {0};
  event_attrib.version               = NVTX_VERSION;
  event_attrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event_attrib.colorType             = NVTX_COLOR_ARGB;
  event_attrib.color                 = generate_next_color(name);
  event_attrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
  event_attrib.message.ascii         = name;
  nvtxDomainRangePushEx(domain_store<Domain>::value(), &event_attrib);
}

template <typename Domain, typename... Args>
inline void push_range(const char* format, Args... args)
{
  if constexpr (sizeof...(args) > 0) {
    int length = std::snprintf(nullptr, 0, format, args...);
    assert(length >= 0);
    std::vector<char> buf(length + 1);
    std::snprintf(buf.data(), length + 1, format, args...);
    push_range_name<Domain>(buf.data());
  } else {
    push_range_name<Domain>(format);
  }
}

template <typename Domain>
inline void pop_range()
{
  nvtxDomainRangePop(domain_store<Domain>::value());
}

#else  // NVTX_ENABLED

template <typename Domain, typename... Args>
inline void push_range(const char* format, Args... args)
{
}

template <typename Domain>
inline void pop_range()
{
}

#endif  // NVTX_ENABLED

}  // namespace raft::common::nvtx::detail
