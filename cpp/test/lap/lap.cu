/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <omp.h>
#include <iostream>
#include <random>
#include "raft/lap/lap.cuh"

#define PROBLEMSIZE 1000  // Number of rows/columns
#define BATCHSIZE 10      // Number of problems in the batch
#define COSTRANGE 1000
#define PROBLEMCOUNT 1
#define REPETITIONS 1

#define SEED 01010001

std::default_random_engine generator(SEED);

// Function for generating problem with uniformly distributed integer costs between [0, COSTRANGE].
template <typename weight_t>
void generateProblem(weight_t *cost_matrix, int SP, int N, int costrange) {
  long N2 = SP * N * N;

  std::uniform_int_distribution<int> distribution(0, costrange);

  for (long i = 0; i < N2; i++) {
    int val = distribution(generator);
    cost_matrix[i] = (weight_t)val;
  }
}

template <typename vertex_t, typename weight_t>
void hungarian_test(int problemsize, int costrange, int problemcount,
                    int repetitions, int batchsize) {
  raft::handle_t handle;

  weight_t *h_cost = new weight_t[batchsize * problemsize * problemsize];

  printf("(%d, %d)\n", problemsize, costrange);

  for (int j = 0; j < problemcount; j++) {
    generateProblem(h_cost, batchsize, problemsize, costrange);

    raft::mr::device::buffer<weight_t> elements_v(
      handle.get_device_allocator(), handle.get_stream(),
      batchsize * problemsize * problemsize);

    raft::update_device(elements_v.data(), h_cost,
                        batchsize * problemsize * problemsize,
                        handle.get_stream());

    for (int i = 0; i < repetitions; i++) {
      float start = omp_get_wtime();

      // Create an instance of LinearAssignmentProblem using problem size, number of subproblems
      raft::lap::LinearAssignmentProblem<vertex_t, weight_t> lpx(
        handle, problemsize, batchsize);

      // Solve LAP(s) for given cost matrix
      lpx.solve(elements_v.data());

      float end = omp_get_wtime();

      float total_time = (end - start);

      // Use getPrimalObjectiveValue and getDualObjectiveValue APIs to get primal and dual objectives. At optimality both values should match.
      for (int k = 0; k < batchsize; k++) {
        std::cout << j << ":" << i << ":" << k << ":"
                  << lpx.getPrimalObjectiveValue(k) << ":"
                  << lpx.getDualObjectiveValue(k) << ":" << total_time
                  << std::endl;
      }

      //			Use getAssignmentVector API to get the optimal row assignments for specified problem id.
      //			Example is shown below.

      //			int *assignment_sp1 = new int[problemsize];
      //			lpx.getAssignmentVector(assignment_sp1, 15);
      //
      //			std::cout << "\nPrinting assignment vector for subproblem 1" << std::endl;
      //			for (int z = 0; z < problemsize; z++) {
      //				std::cout << z << "\t" << assignment_sp1[z] << std::endl;
      //			}
      //
      //			delete[] assignment_sp1;

      //			Use getRowDualVector and getColDualVector API to get the optimal row duals and column duals for specified problem id.
      //			Example is shown below.

      //			float *row_dual_sp1 = new float[problemsize];
      //			lpx.getRowDualVector(row_dual_sp1, 15);
      //
      //			std::cout << "\nPrinting row dual vector for subproblem 1" << std::endl;
      //			for (int z = 0; z < problemsize; z++) {
      //				std::cout << z << "\t" << row_dual_sp1[z] << std::endl;
      //			}
      //
      //			delete[] row_dual_sp1;
    }
  }

  delete[] h_cost;
}

TEST(Raft, HungarianIntFloat) {
  hungarian_test<int, float>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS,
                             BATCHSIZE);
}

TEST(Raft, HungarianIntDouble) {
  hungarian_test<int, double>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS,
                              BATCHSIZE);
}

TEST(Raft, HungarianIntLong) {
  hungarian_test<int, long>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS,
                            BATCHSIZE);
}

TEST(Raft, HungarianLongFloat) {
  hungarian_test<long, float>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS,
                              BATCHSIZE);
}

TEST(Raft, HungarianLongDouble) {
  hungarian_test<long, double>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT,
                               REPETITIONS, BATCHSIZE);
}

TEST(Raft, HungarianLongLong) {
  hungarian_test<long, long>(PROBLEMSIZE, COSTRANGE, PROBLEMCOUNT, REPETITIONS,
                             BATCHSIZE);
}
