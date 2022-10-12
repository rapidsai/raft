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

#include <common/benchmark.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct mat_vec_op_inputs {
  int rows, cols;
  bool rowMajor, bcastAlongRows;
};  // struct mat_vec_op_inputs

template <typename T>
struct mat_vec_op : public fixture {
  mat_vec_op(const mat_vec_op_inputs& p)
    : params(p),
      out(p.rows * p.cols, stream),
      in(p.rows * p.cols, stream),
      vec(p.bcastAlongRows ? p.cols : p.rows, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::matrixVectorOp(out.data(),
                                   in.data(),
                                   vec.data(),
                                   params.cols,
                                   params.rows,
                                   params.rowMajor,
                                   params.bcastAlongRows,
                                   raft::Sum<T>(),
                                   stream);
    });
  }

 private:
  mat_vec_op_inputs params;
  rmm::device_uvector<T> out, in, vec;
};  // struct MatVecOp

const std::vector<mat_vec_op_inputs> mat_vec_op_input_vecs{
  {1024, 128, true, true},       {1024 * 1024, 128, true, true},
  {1024, 128 + 2, true, true},   {1024 * 1024, 128 + 2, true, true},
  {1024, 128 + 1, true, true},   {1024 * 1024, 128 + 1, true, true},

  {1024, 128, true, false},      {1024 * 1024, 128, true, false},
  {1024, 128 + 2, true, false},  {1024 * 1024, 128 + 2, true, false},
  {1024, 128 + 1, true, false},  {1024 * 1024, 128 + 1, true, false},

  {1024, 128, false, false},     {1024 * 1024, 128, false, false},
  {1024, 128 + 2, false, false}, {1024 * 1024, 128 + 2, false, false},
  {1024, 128 + 1, false, false}, {1024 * 1024, 128 + 1, false, false},

  {1024, 128, false, true},      {1024 * 1024, 128, false, true},
  {1024, 128 + 2, false, true},  {1024 * 1024, 128 + 2, false, true},
  {1024, 128 + 1, false, true},  {1024 * 1024, 128 + 1, false, true},

};

RAFT_BENCH_REGISTER(mat_vec_op<float>, "", mat_vec_op_input_vecs);
RAFT_BENCH_REGISTER(mat_vec_op<double>, "", mat_vec_op_input_vecs);

}  // namespace raft::bench::linalg
