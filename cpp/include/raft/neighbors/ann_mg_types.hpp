/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/ivf_pq_types.hpp>
#include <raft/neighbors/cagra_types.hpp>

namespace raft::neighbors::mg {
  enum parallel_mode { REPLICATION, SHARDING };
}

namespace raft::neighbors::ivf_flat {
  struct mg_index_params : raft::neighbors::ivf_flat::index_params {
    raft::neighbors::mg::parallel_mode mode;
  };
}

namespace raft::neighbors::ivf_pq {
  struct mg_index_params : raft::neighbors::ivf_pq::index_params {
    raft::neighbors::mg::parallel_mode mode;
  };
}

namespace raft::neighbors::cagra {
  struct mg_index_params : raft::neighbors::cagra::index_params {
    raft::neighbors::mg::parallel_mode mode;
  };
}
