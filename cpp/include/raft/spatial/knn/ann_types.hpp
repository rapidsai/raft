/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/distance/distance_types.hpp>

namespace raft::spatial::knn {

/** The base for approximate KNN index structures. */
struct index {};

/** The base for KNN index parameters. */
struct index_params {
  /** Distance type. */
  raft::distance::DistanceType metric = distance::DistanceType::L2Expanded;
  /** The argument used by some distance metrics. */
  float metric_arg = 2.0f;
  /**
   * Whether to add the dataset content to the index, i.e.:
   *
   *  - `true` means the index is filled with the dataset vectors and ready to search after calling
   * `build`.
   *  - `false` means `build` only trains the underlying model (e.g. quantizer or clustering), but
   * the index is left empty; you'd need to call `extend` on the index afterwards to populate it.
   */
  bool add_data_on_build = true;
};

struct search_params {};

};  // namespace raft::spatial::knn
