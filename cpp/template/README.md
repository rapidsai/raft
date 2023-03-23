# Example RAFT Project Template

This template project provides a drop-in sample to either start building a new application with, or using RAFT in an existing CMake project. 

First, please refer to our [installation docs](https://docs.rapids.ai/api/raft/stable/build.html#cuda-gpu-requirements) for the minimum requirements to use RAFT.

Once the minimum requirements are satisfied, this example template application can be built with the provided `build.sh` script. This is a bash script that calls the appropriate CMake commands, so you can look into it to see the typical CMake based build workflow.  

This directory (`RAFT_SOURCE/cpp/template`) can be copied directly in order to build a new application with RAFT.

RAFT can be integrated into an existing CMake project by copying the contents in the `configure rapids-cmake` and `configure raft` sections of the provided `CMakeLists.txt` into your project, along with `cmake/thirdparty/get_raft.cmake`. 

Make sure to link against the appropriate Cmake targets. Use `raft::raft`to add make the headers available and `raft::compiled` when utilizing the shared library.

```cmake
target_link_libraries(your_app_target PRIVATE raft::raft raft::compiled)
```

