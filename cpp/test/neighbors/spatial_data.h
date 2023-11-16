/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <vector>

namespace raft {
namespace spatial {

// Latitude and longitude coordinates of 51 US states / territories
std::vector<float> spatial_data = {
  63.588753, -154.493062, 32.318231, -86.902298,  35.20105,  -91.831833,  34.048928, -111.093731,
  36.778261, -119.417932, 39.550051, -105.782067, 41.603221, -73.087749,  38.905985, -77.033418,
  38.910832, -75.52767,   27.664827, -81.515754,  32.157435, -82.907123,  19.898682, -155.665857,
  41.878003, -93.097702,  44.068202, -114.742041, 40.633125, -89.398528,  40.551217, -85.602364,
  39.011902, -98.484246,  37.839333, -84.270018,  31.244823, -92.145024,  42.407211, -71.382437,
  39.045755, -76.641271,  45.253783, -69.445469,  44.314844, -85.602364,  46.729553, -94.6859,
  37.964253, -91.831833,  32.354668, -89.398528,  46.879682, -110.362566, 35.759573, -79.0193,
  47.551493, -101.002012, 41.492537, -99.901813,  43.193852, -71.572395,  40.058324, -74.405661,
  34.97273,  -105.032363, 38.80261,  -116.419389, 43.299428, -74.217933,  40.417287, -82.907123,
  35.007752, -97.092877,  43.804133, -120.554201, 41.203322, -77.194525,  18.220833, -66.590149,
  41.580095, -71.477429,  33.836081, -81.163725,  43.969515, -99.901813,  35.517491, -86.580447,
  31.968599, -99.901813,  39.32098,  -111.093731, 37.431573, -78.656894,  44.558803, -72.577841,
  47.751074, -120.740139, 43.78444,  -88.787868,  38.597626, -80.454903,  43.075968, -107.290284};
};  // namespace spatial
};  // namespace raft