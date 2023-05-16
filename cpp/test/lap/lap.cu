/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * Copyright 2020 KETAN DATE & RAKESH NAGI
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
 *
 *      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
 *      Authors: Ketan Date and Rakesh Nagi
 *
 *      Article reference:
 *          Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms
 *          for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
 *
 */
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <iostream>
#include <omp.h>
#include <raft/solver/linear_assignment.cuh>
#include <random>

#define PROBLEMSIZE  1000  // Number of rows/columns
#define BATCHSIZE    10    // Number of problems in the batch
#define COSTRANGE    1000
#define PROBLEMCOUNT 1
#define REPETITIONS  1

#define SEED 01010001

std::default_random_engine generator(SEED);

namespace raft {

// Function for generating problem with uniformly distributed integer costs between [0, COSTRANGE].
template <typename weight_t>
void generateProblem(weight_t* cost_matrix, int SP, int N, int costrange)
{
  long N2 = SP * N * N;

  std::uniform_int_distribution<int> distribution(0, costrange);

  for (long i = 0; i < N2; i++) {
    int val        = distribution(generator);
    cost_matrix[i] = (weight_t)val;
  }
}

template <typename vertex_t, typename weight_t>
void hungarian_test(int problemsize,
                    int costrange,
                    int problemcount,
                    int repetitions,
                    int batchsize,
                    weight_t epsilon,
                    bool verbose = false)
{
  raft::resources handle;

  weight_t* h_cost = new weight_t[batchsize * problemsize * problemsize];

  for (int j = 0; j < problemcount; j++) {
    generateProblem(h_cost, batchsize, problemsize, costrange);

    rmm::device_uvector<weight_t> elements_v(batchsize * problemsize * problemsize,
                                             resource::get_cuda_stream(handle));
    rmm::device_uvector<vertex_t> row_assignment_v(batchsize * problemsize,
                                                   resource::get_cuda_stream(handle));
    rmm::device_uvector<vertex_t> col_assignment_v(batchsize * problemsize,
                                                   resource::get_cuda_stream(handle));

    raft::update_device(elements_v.data(),
                        h_cost,
                        batchsize * problemsize * problemsize,
                        resource::get_cuda_stream(handle));

    for (int i = 0; i < repetitions; i++) {
      float start = omp_get_wtime();

      // Create an instance of LinearAssignmentProblem using problem size, number of subproblems
      raft::solver::LinearAssignmentProblem<vertex_t, weight_t> lpx(
        handle, problemsize, batchsize, epsilon);

      // Solve LAP(s) for given cost matrix
      lpx.solve(elements_v.data(), row_assignment_v.data(), col_assignment_v.data());

      float end = omp_get_wtime();

      float total_time = (end - start);

      if (verbose) {
        // Use getPrimalObjectiveValue and getDualObjectiveValue APIs to get primal and dual
        // objectives. At optimality both values should match.
        for (int k = 0; k < batchsize; k++) {
          std::cout << j << ":" << i << ":" << k << ":" << lpx.getPrimalObjectiveValue(k) << ":"
                    << lpx.getDualObjectiveValue(k) << ":" << total_time << std::endl;
        }
      }
    }
  }

  delete[] h_cost;
}

TEST(Raft, HungarianIntFloat)
{
  hungarian_test<int, float>(
    PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS, BATCHSIZE, float{1e-6});
}

TEST(Raft, HungarianIntDouble)
{
  hungarian_test<int, double>(
    PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS, BATCHSIZE, double{1e-6});
}

TEST(Raft, HungarianIntLong)
{
  hungarian_test<int, long>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS, BATCHSIZE, long{0});
}

TEST(Raft, HungarianLongFloat)
{
  hungarian_test<long, float>(
    PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS, BATCHSIZE, float{1e-6});
}

TEST(Raft, HungarianLongDouble)
{
  hungarian_test<long, double>(
    PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS, BATCHSIZE, double{1e-6});
}

TEST(Raft, HungarianLongLong)
{
  hungarian_test<long, long>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS, BATCHSIZE, long{0});
}

}  // namespace raft
