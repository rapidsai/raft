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
