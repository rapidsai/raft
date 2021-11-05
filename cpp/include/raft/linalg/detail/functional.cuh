/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

 #include <thrust/functional.h>

 namespace raft {
 namespace linalg {
 namespace detail {

template <typename ArgType, typename ReturnType = ArgType>
struct divides_scalar {

public:
    divides_scalar(ArgType scalar) : scalar_(scalar) {} 

    __host__ __device__ inline ReturnType operator()(ArgType in) {
        return in / scalar_;
    }

private:
    ArgType scalar_;
};

template <typename ArgType, typename ReturnType = ArgType>
struct adds_scalar {

public:
    adds_scalar(ArgType scalar) : scalar_(scalar) {} 

    __host__ __device__ inline ReturnType operator()(ArgType in) {
        return in + scalar_;
    }

private:
    ArgType scalar_;
};

template <typename ArgType, typename ReturnType = ArgType>
struct multiplies_scalar {

public:
    multiplies_scalar(ArgType scalar) : scalar_(scalar) {} 

    __host__ __device__ inline ReturnType operator()(ArgType in) {
        return in * scalar_;
    }

private:
    ArgType scalar_;
};

template <typename ArgType, typename ReturnType = ArgType>
struct divides_check_zero {

public:
    __host__ __device__ inline ReturnType operator()(ArgType a, ArgType b) {
        return (b == static_cast<ArgType>(0)) ? 0.0 : a / b;
    }

};


} // namespace detail
} // namespace linalg
} // namespace raft