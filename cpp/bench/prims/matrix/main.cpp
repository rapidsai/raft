/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>

#include <cstring>

namespace raft::matrix {
void add_select_k_dataset_benchmarks();
}

int main(int argc, char** argv)
{
  // if we're passed a 'select_k_dataset' flag, add in extra benchmarks
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--select_k_dataset") == 0) {
      raft::matrix::add_select_k_dataset_benchmarks();

      // pop off the cmdline argument from argc/argv
      for (int j = i; j < argc - 1; ++j)
        argv[j] = argv[j + 1];
      argc--;
      break;
    }
  }
  benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
}
