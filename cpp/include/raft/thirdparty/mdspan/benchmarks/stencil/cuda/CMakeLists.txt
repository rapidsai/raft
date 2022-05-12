
mdspan_add_cuda_benchmark(stencil_3d_cuda)
target_include_directories(stencil_3d_cuda PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks/stencil>
)
