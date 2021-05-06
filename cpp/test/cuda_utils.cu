#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <raft/cuda_utils.cuh>

void __global__ testWarpReduce(float* data, float* out) {
  size_t half_warp = raft::warp_size() / 2;
  uint32_t mask = __ballot_sync(raft::warp_full_mask(), raft::laneId() < half_warp);
  float val = 0.0f;
  if (raft::laneId() < half_warp) {
    val = raft::warpReduce(data[threadIdx.x], mask);
  }

  if (raft::laneId() == 0) {
    out[raft::warpId()] = val;
  }
}

TEST(Raft, WarpReduce) {
  size_t constexpr kBlock = 256, kGrid = 2;
  thrust::device_vector<float> input(kBlock * kGrid);
  thrust::fill(input.begin(), input.end(), 1.0f);

  // allocate buffer with size equal to number of warps
  thrust::device_vector<float> out(input.size() / raft::warp_size());

  testWarpReduce<<<kGrid, kBlock>>>(input.data().get(), out.data().get());
  cudaDeviceSynchronize();

  for (size_t i = 0; i < out.size(); ++i) {
    ASSERT_EQ(out[i], float(raft::warp_size() / 2));
  }
}
