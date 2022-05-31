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

#pragma once

#include <raft/core/handle.hpp>
#include <raft/sparse/detail/mask.hpp>
#include <rmm/device_uvector.hpp>
#include <cstdint>
#include <optional>
#include <vector>

namespace raft {

/**
 * An owning container object to manage separate bitmasks for
 * filtering vertices and edges. A compliment setting
 * determines whether the value of 1 for corresponding
 * items means they should be masked in (included) or
 * masked out (excluded).
 *
 * @tparam vertex_t
 * @tparam edge_t
 * @tparam mask_t
 */
template <typename vertex_t, typename edge_t, typename mask_t = std::uint32_t>
struct mask {
public:
    mask() = delete;

    mask(raft::handle_t const &handle,
         vertex_t n_vertices,
         edge_t n_edges,
         bool complement = false) :
            n_vertices_(n_vertices),
            n_edges_(n_edges),
            edges_(0, handle.get_stream()),
            vertices_(0, handle.get_stream()),
            complement_(complement)
    {}

    bool is_complemented() const { return complement_; }

    bool has_edge_mask() const { return get_edge_mask().has_value(); }
    bool has_vertex_mask() const { return get_vertex_mask().has_value(); }

    std::optional<mask_t const*> get_edge_mask() const {
        return edges_.size() > 0 ? std::make_optional<mask_t const*> edges_.data() : std::nullopt;
    }

    std::optional<mask_t const*> get_vertex_mask() const {
        return vertices_.size() > 0 ? std::make_optional<mask_t const*> vertices_.data() : std::nullopt;
    }

    void initialize_edge_mask() {
        if(edges_.size() == 0) {
            edges_.resize(n_edges_, handle.get_stream());
        }
    }

    void initialize_vertex_mask() {
        if(vertices_.size() == 0) {
            vertices_.resize(n_vertices_, handle.get_stream());
        }
    }

private:
    raft::handle_t const &handle;
    vertex_t n_vertices_;
    edge_t n_edges_;
    bool complement_ = false;
    rmm::device_uvector<mask_t> vertices_;
    rmm::device_uvector<mask_t> edges_;
};
}

