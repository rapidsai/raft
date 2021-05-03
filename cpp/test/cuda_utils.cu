#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <raft/cuda_utils.cuh>

void __global__ testWarpReduce(float* data, float* out) {
  uint32_t mask = __ballot_sync(raft::warp_full_mask(), raft::laneId() < 16);
  float val = 0.0f;
  if (raft::laneId() < 16) {
    val = raft::warpReduce(data[threadIdx.x], mask);
  }

  data[raft::warpId()] = val;
}


TEST(Raft, WarpReduce) {
  size_t constexpr kBlock = 256;
  thrust::device_vector<float> buffer(kBlock);
  thrust::fill(buffer.begin(), buffer.end(), 1.0f);

  thrust::device_vector<float> out(buffer.size() / raft::warp_size());

  testWarpReduce<<<1, kBlock>>>(buffer.data().get(), out.data().get());
  cudaDeviceSynchronize();

  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i], float(raft::warp_size() / 2));
  }
}
