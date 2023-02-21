# Developer Guide

## General
Please start by reading the [Contributor Guide](contributing.md).

## Performance
1. In performance critical sections of the code, favor `cudaDeviceGetAttribute` over `cudaDeviceGetProperties`. See corresponding CUDA devblog [here](https://devblogs.nvidia.com/cuda-pro-tip-the-fast-way-to-query-device-properties/) to know more.
2. If an algo requires you to launch GPU work in multiple cuda streams, do not create multiple `raft::resources` objects, one for each such work stream. Instead, use the stream pool configured on the given `raft::resources` instance's `raft::resources::get_stream_from_stream_pool()` to pick up the right cuda stream. Refer to the section on [CUDA Resources](#resource-management) and the section on [Threading](#threading-model) for more details. TIP: use `raft::resources::get_stream_pool_size()` to know how many such streams are available at your disposal.


## Local Development

Developing features and fixing bugs for the RAFT library itself is straightforward and only requires building and installing the relevant RAFT artifacts.

The process for working on a CUDA/C++ feature which might span RAFT and one or more consuming libraries can vary slightly depending on whether the consuming project relies on a source build (as outlined in the [BUILD](BUILD.md#install_header_only_cpp) docs). In such a case, the option `CPM_raft_SOURCE=/path/to/raft/source` can be passed to the cmake of the consuming project in order to build the local RAFT from source. The PR with relevant changes to the consuming project can also pin the RAFT version temporarily by explicitly changing the `FORK` and `PINNED_TAG` arguments to the RAFT branch containing their changes when invoking `find_and_configure_raft`.  The pin should be reverted after the changed is merged to the RAFT project and before it is merged to the dependent project(s) downstream.

If building a feature which spans projects and not using the source build in cmake, the RAFT changes (both C++ and Python) will need to be installed into the environment of the consuming project before they can be used. The ideal integration of RAFT into consuming projects will enable both the source build in the consuming project only for this case but also rely on a more stable packaging (such as conda packaging) otherwise. 


## Threading Model

With the exception of the `raft::resources`, RAFT algorithms should maintain thread-safety and are, in general,
assumed to be single threaded. This means they should be able to be called from multiple host threads so
long as different instances of `raft::resources` are used.

Exceptions are made for algorithms that can take advantage of multiple CUDA streams within multiple host threads
in order to oversubscribe or increase occupancy on a single GPU. In these cases, the use of multiple host
threads within RAFT algorithms should be used only to maintain concurrency of the underlying CUDA streams.
Multiple host threads should be used sparingly, be bounded, and should steer clear of performing CPU-intensive
computations.

A good example of an acceptable use of host threads within a RAFT algorithm might look like the following

```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
raft::resources res;

...

sync_stream(res);

...

int n_streams = get_stream_pool_size(res);

#pragma omp parallel for num_threads(n_threads)
for(int i = 0; i < n; i++) {
    int thread_num = omp_get_thread_num() % n_threads;
    cudaStream_t s = get_stream_from_stream_pool(res, thread_num);
    ... possible light cpu pre-processing ...
    my_kernel1<<<b, tpb, 0, s>>>(...);
    ...
    ... some possible async d2h / h2d copies ...
    my_kernel2<<<b, tpb, 0, s>>>(...);
    ...
    sync_stream(res, s);
    ... possible light cpu post-processing ...
}
```

In the example above, if there is no CPU pre-processing at the beginning of the for-loop, an event can be registered in
each of the streams within the for-loop to make them wait on the stream from the handle. If there is no CPU post-processing
at the end of each for-loop iteration, `sync_stream(res, s)` can be replaced with a single `sync_stream_pool(res)`
after the for-loop.

To avoid compatibility issues between different threading models, the only threading programming allowed in RAFT is OpenMP.
Though RAFT's build enables OpenMP by default, RAFT algorithms should still function properly even when OpenMP has been
disabled. If the CPU pre- and post-processing were not needed in the example above, OpenMP would not be needed.

The use of threads in third-party libraries is allowed, though they should still avoid depending on a specific OpenMP runtime.

## Public Interface

### General guidelines
Functions exposed via the C++ API must be stateless. Things that are OK to be exposed on the interface:
1. Any [POD](https://en.wikipedia.org/wiki/Passive_data_structure) - see [std::is_pod](https://en.cppreference.com/w/cpp/types/is_pod) as a reference for C++11  POD types.
2. `raft::resources` - since it stores resource-related state which has nothing to do with model/algo state.
3. Avoid using pointers to POD types (explicitly putting it out, even though it can be considered as a POD) and pass the structures by reference instead.
   Internal to the C++ API, these stateless functions are free to use their own temporary classes, as long as they are not exposed on the interface.
4. Accept single- (`raft::span`) and multi-dimensional views (`raft::mdspan`) and validate their metadata wherever possible.
5. Prefer `std::optional` for any optional arguments (e.g. do not accept `nullptr`)
6. All public APIs should be lightweight wrappers around calls to private APIs inside the `detail` namespace.

### API stability

Since RAFT is a core library with multiple consumers, it's important that the public APIs maintain stability across versions and any changes to them are done with caution, adding new functions and deprecating the old functions over a couple releases as necessary.

### Stateless C++ APIs

Using the IVF-PQ algorithm as an example, the following way of exposing its API would be wrong according to the guidelines in this section, since it exposes a non-POD C++ class object in the C++ API:
```cpp
template <typename value_t, typename idx_t>
class ivf_pq {
  ivf_pq_params params_;
  raft::resources const& res_;
  
public:
  ivf_pq(raft::resources const& res);
  void train(raft::device_matrix<value_t, idx_t, raft::row_major> dataset);
  void search(raft::device_matrix<value_t, idx_t, raft::row_major> queries, 
              raft::device_matrix<value_t, idx_t, raft::row_major> out_inds, 
              raft::device_matrix<value_t, idx_t, raft::row_major> out_dists);
};
```

An alternative correct way to expose this could be:
```cpp
namespace raft::ivf_pq {

template<typename value_t, typename value_idx>
void ivf_pq_train(raft::resources const& res, const raft::ivf_pq_params &params, raft::ivf_pq_index &index,
raft::device_matrix<value_t, idx_t, raft::row_major> dataset);

template<typename value_t, typename value_idx>
void ivf_pq_search(raft::resources const& res, raft::ivf_pq_params const&params, raft::ivf_pq_index const & index,
raft::device_matrix<value_t, idx_t, raft::row_major> queries,
raft::device_matrix<value_t, idx_t, raft::row_major> out_inds,
raft::device_matrix<value_t, idx_t, raft::row_major> out_dists);
}
```

### Other functions on state

These guidelines also mean that it is the responsibility of C++ API to expose methods to load and store (aka marshalling) such a data structure. Further continuing the IVF-PQ example,  the following methods could achieve this:
```cpp
namespace raft::ivf_pq {
   void save(raft::ivf_pq_index const& model, std::ostream &os);
   void load(raft::ivf_pq_index& model, std::istream &is);
}
```


## Coding style

### Code format
#### Introduction
RAFT relies on `clang-format` to enforce code style across all C++ and CUDA source code. The coding style is based on the [Google style guide](https://google.github.io/styleguide/cppguide.html#Formatting). The only digressions from this style are the following.
1. Do not split empty functions/records/namespaces.
2. Two-space indentation everywhere, including the line continuations.
3. Disable reflowing of comments.
   The reasons behind these deviations from the Google style guide are given in comments [here](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/.clang-format).

#### How is the check done?
All formatting checks are done by this python script: [run-clang-format.py](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/scripts/run-clang-format.py) which is effectively a wrapper over `clang-format`. An error is raised if the code diverges from the format suggested by clang-format. It is expected that the developers run this script to detect and fix formatting violations before creating PR.

##### As part of CI
[run-clang-format.py](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/scripts/run-clang-format.py) is executed as part of our `ci/checks/style.sh` CI test. If there are any formatting violations, PR author is expected to fix those to get CI passing. Steps needed to fix the formatting violations are described in the subsequent sub-section.

##### Manually
Developers can also manually (or setup this command as part of git pre-commit hook) run this check by executing:
```bash
python ./cpp/scripts/run-clang-format.py
```
From the root of the RAFT repository.

#### How to know the formatting violations?
When there are formatting errors, [run-clang-format.py](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/scripts/run-clang-format.py) prints a `diff` command, showing where there are formatting differences. Unfortunately, unlike `flake8`, `clang-format` does NOT print descriptions of the violations, but instead directly formats the code. So, the only way currently to know about formatting differences is to run the diff command as suggested by this script against each violating source file.

#### How to fix the formatting violations?
When there are formatting violations, [run-clang-format.py](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/scripts/run-clang-format.py) prints at the end, the exact command that can be run by developers to fix them. This is the easiest way to fix formatting errors. [This screencast](https://asciinema.org/a/287367) shows how developers can check for formatting violations in their branches and also how to fix those, before sending out PRs.

In short, to bulk-fix all the formatting violations, execute the following command:
```bash
python ./cpp/scripts/run-clang-format.py -inplace
```
From the root of the RAFT repository.

#### clang-format version?
To avoid spurious code style violations we specify the exact clang-format version required, currently `11.1.0`. This is enforced by the [run-clang-format.py](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/scripts/run-clang-format.py) script itself. Refer [here](../build#build-dependencies) for the list of build-time dependencies.

#### Additional scripts
Along with clang, there are an include checker and copyright checker scripts for checking style, which can be performed as part of CI, as well as manually.

##### #include style
[include_checker.py](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/scripts/include_checker.py) is used to enforce the include style as follows:
1. `#include "..."` should be used for referencing local files only. It is acceptable to be used for referencing files in a sub-folder/parent-folder of the same algorithm, but should never be used to include files in other algorithms or between algorithms and the primitives or other dependencies.
2. `#include <...>` should be used for referencing everything else

Manually, run the following to bulk-fix include style issues:
```bash
python ./cpp/scripts/include_checker.py --inplace [cpp/include cpp/test ... list of folders which you want to fix]
```

##### Copyright header
[copyright.py](https://github.com/rapidsai/raft/blob/branch-23.04/ci/checks/copyright.py) checks the Copyright header for all git-modified files

Manually, you can run the following to bulk-fix the header if only the years need to be updated:
```bash
python ./ci/checks/copyright.py --update-current-year
```
Keep in mind that this only applies to files tracked by git and having been modified.

## Error handling
Call CUDA APIs via the provided helper macros `RAFT_CUDA_TRY`, `RAFT_CUBLAS_TRY` and `RAFT_CUSOLVER_TRY`. These macros take care of checking the return values of the used API calls and generate an exception when the command is not successful. If you need to avoid an exception, e.g. inside a destructor, use `RAFT_CUDA_TRY_NO_THROW`, `RAFT_CUBLAS_TRY_NO_THROW ` and `RAFT_CUSOLVER_TRY_NO_THROW`. These macros log the error but do not throw an exception.

## Logging

### Introduction
Anything and everything about logging is defined inside [logger.hpp](https://github.com/rapidsai/raft/blob/branch-23.04/cpp/include/raft/core/logger.hpp). It uses [spdlog](https://github.com/gabime/spdlog) underneath, but this information is transparent to all.

### Usage
```cpp
#include <raft/core/logger.hpp>

// Inside your method or function, use any of these macros
RAFT_LOG_TRACE("Hello %s!", "world");
RAFT_LOG_DEBUG("Hello %s!", "world");
RAFT_LOG_INFO("Hello %s!", "world");
RAFT_LOG_WARN("Hello %s!", "world");
RAFT_LOG_ERROR("Hello %s!", "world");
RAFT_LOG_CRITICAL("Hello %s!", "world");
```

### Changing logging level
There are 7 logging levels with each successive level becoming quieter:
1. RAFT_LEVEL_TRACE
2. RAFT_LEVEL_DEBUG
3. RAFT_LEVEL_INFO
4. RAFT_LEVEL_WARN
5. RAFT_LEVEL_ERROR
6. RAFT_LEVEL_CRITICAL
7. RAFT_LEVEL_OFF
   Pass one of these as per your needs into the `set_level()` method as follows:
```cpp
raft::logger::get.set_level(RAFT_LEVEL_WARN);
// From now onwards, this will print only WARN and above kind of messages
```

### Changing logging pattern
Pass the [format string](https://github.com/gabime/spdlog/wiki/3.-Custom-formatting) as follows in order use a different logging pattern than the default.
```cpp
raft::logger::get.set_pattern(YourFavoriteFormat);
```
One can also use the corresponding `get_pattern()` method to know the current format as well.

### Temporarily changing the logging pattern
Sometimes, we need to temporarily change the log pattern (eg: for reporting decision tree structure). This can be achieved in a RAII-like approach as follows:
```cpp
{
  PatternSetter _(MyNewTempFormat);
  // new log format is in effect from here onwards
  doStuff();
  // once the above temporary object goes out-of-scope, the old format will be restored
}
```

### Tips
* Do NOT end your logging messages with a newline! It is automatically added by spdlog.
* The `RAFT_LOG_TRACE()` is by default not compiled due to the `RAFT_ACTIVE_LEVEL` macro setup, for performance reasons. If you need it to be enabled, change this macro accordingly during compilation time

## Common Design Considerations

1. Use the `hpp` extension for files which can be compiled with `gcc` against the CUDA-runtime. Use the `cuh` extension for files which require `nvcc` to be compiled. `hpp` can also be used for functions marked `__host__ __device__` only if proper checks are in place to remove the `__device__` designation when not compiling with `nvcc`. 

2. When additional classes, structs, or general POCO types are needed to be used for representing data in the public API, place them in a new file called `<primitive_name>_types.hpp`. This tells users they are safe to expose these types on their own public APIs without bringing in device code. At a minimum, the definitions for these types, at least, should not require `nvcc`. In general, these classes should only store very simple state and should not perform their own computations. Instead, new functions should be exposed on the public API which accept these objects, reading or updating their state as necessary. 

3. Documentation for public APIs should be well documented, easy to use, and it is highly preferred that they include usage instructions.

4. Before creating a new primitive, check to see if one exists already. If one exists but the API isn't flexible enough to include your use-case, consider first refactoring the existing primitive. If that is not possible without an extreme number of changes, consider how the public API could be made more flexible. If the new primitive is different enough from all existing primitives, consider whether an existing public API could invoke the new primitive as an option or argument. If the new primitive is different enough from what exists already, add a header for the new public API function to the appropriate subdirectory and namespace.

## Testing

It's important for RAFT to maintain a high test coverage of the public APIs in order to minimize the potential for downstream projects to encounter unexpected build or runtime behavior as a result of changes. 

A well-defined public API can help maintain compile-time stability but means more focus should be placed on testing the functional requirements and verifying execution on the various edge cases within RAFT itself. Ideally, bug fixes and new features should be able to be made to RAFT independently of the consuming projects.

## Documentation

Public APIs always require documentation since those will be exposed directly to users. For C++, we use [doxygen](http://www.doxygen.nl) and for Python/cython we use [pydoc](https://docs.python.org/3/library/pydoc.html). In addition to summarizing the purpose of each class / function in the public API, the arguments (and relevant templates) should be documented along with brief usage examples.

## Asynchronous operations and stream ordering
All RAFT algorithms should be as asynchronous as possible avoiding the use of the default stream (aka as NULL or `0` stream). Implementations that require only one CUDA Stream should use the stream from `raft::resources`:

```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

void foo(const raft::resources& res, ...)
{
    cudaStream_t stream = get_cuda_stream(res);
}
```
When multiple streams are needed, e.g. to manage a pipeline, use the internal streams available in `raft::resources` (see [CUDA Resources](#cuda-resources)). If multiple streams are used all operations still must be ordered according to `raft::resource::get_cuda_stream()` (from `raft/core/resource/cuda_stream.hpp`). Before any operation in any of the internal CUDA streams is started, all previous work in `raft::resource::get_cuda_stream()` must have completed. Any work enqueued in `raft::resource::get_cuda_stream()` after a RAFT function returns should not start before all work enqueued in the internal streams has completed. E.g. if a RAFT algorithm is called like this:
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
void foo(const double* srcdata, double* result)
{
    cudaStream_t stream;
    CUDA_RT_CALL( cudaStreamCreate( &stream ) );
    raft::resources res;
    set_cuda_stream(res, stream);
    
    ...
    
    RAFT_CUDA_TRY( cudaMemcpyAsync( srcdata, h_srcdata.data(), n*sizeof(double), cudaMemcpyHostToDevice, stream ) );

    raft::algo(raft::resources, dopredict, srcdata, result, ... );

    RAFT_CUDA_TRY( cudaMemcpyAsync( h_result.data(), result, m*sizeof(int), cudaMemcpyDeviceToHost, stream ) );

    ...
}
```
No work in any stream should start in `raft::algo` before the `cudaMemcpyAsync` in `stream` launched before the call to `raft::algo` is done. And all work in all streams used in `raft::algo` should be done before the `cudaMemcpyAsync` in `stream` launched after the call to `raft::algo` starts.

This can be ensured by introducing interstream dependencies with CUDA events and `cudaStreamWaitEvent`. For convenience, the header `raft/core/device_resources.hpp` provides the class `raft::stream_syncer` which lets all `raft::resources` internal CUDA streams wait on `raft::resource::get_cuda_stream()` in its constructor and in its destructor and lets `raft::resource::get_cuda_stream()` wait on all work enqueued in the `raft::resources` internal CUDA streams. The intended use would be to create a `raft::stream_syncer` object as the first thing in an entry function of the public RAFT API:

```cpp
namespace raft {
   void algo(const raft::resources& res, ...)
   {
       raft::streamSyncer _(res);
   }
}
```
This ensures the stream ordering behavior described above.

### Using Thrust
To ensure that thrust algorithms are executed in the intended stream the `thrust::cuda::par` execution policy should be used. To ensure that thrust algorithms allocate temporary memory via the provided device memory allocator, use the `rmm::exec_policy` available in `raft/core/resource/thrust_policy.hpp`, which can be used through `raft::resources`:
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
void foo(const raft::resources& res, ...)
{
    auto execution_policy = get_thrust_policy(res);
    thrust::for_each(execution_policy, ... );
}
```

## Resource Management

Do not create reusable CUDA resources directly in implementations of RAFT algorithms. Instead, use the existing resources in `raft::resources` to avoid constant creation and deletion of reusable resources such as CUDA streams, CUDA events or library handles. Please file a feature request if a resource handle is missing in `raft::resources`.
The resources can be obtained like this
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
void foo(const raft::resources& h, ...)
{
    cublasHandle_t cublasHandle = get_cublas_handle(h);
    const int num_streams       = get_stream_pool_size(h);
    const int stream_idx        = ...
    cudaStream_t stream         = get_stream_from_stream_pool(stream_idx);
    ...
}
```

The example below shows one way to create `n_stream` number of internal cuda streams with an `rmm::stream_pool` which can later be used by the algos inside RAFT. 
```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_pool.hpp>
int main(int argc, char** argv)
{
    int n_streams = argc > 1 ? atoi(argv[1]) : 0;
    raft::resources res;
    set_cuda_stream_pool(res, std::make_shared<rmm::cuda_stream_pool>(n_streams));

    foo(res, ...);
}
```

## Multi-GPU

The multi-GPU paradigm of RAFT is **O**ne **P**rocess per **G**PU (OPG). Each algorithm should be implemented in a way that it can run with a single GPU without any specific dependencies to a particular communication library. A multi-GPU implementation should use the methods offered by the class `raft::comms::comms_t` from [raft/core/comms.hpp] for inter-rank/GPU communication. It is the responsibility of the user of cuML to create an initialized instance of `raft::comms::comms_t`.

E.g. with a CUDA-aware MPI, a RAFT user could use code like this to inject an initialized instance of `raft::comms::mpi_comms` into a `raft::resources`:

```cpp
#include <mpi.h>
#include <raft/core/device_resources.hpp>
#include <raft/comms/mpi_comms.hpp>
#include <raft/algo.hpp>
...
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);

        MPI_Comm_rank(local_comm, &local_rank);

        MPI_Comm_free(&local_comm);
    }

    cudaSetDevice(local_rank);

    mpi_comms raft_mpi_comms;
    MPI_Comm_dup(MPI_COMM_WORLD, &raft_mpi_comms);

    {
        raft::device_resources res;
        initialize_mpi_comms(res, raft_mpi_comms);

        ...

        raft::algo(res, ... );
    }

    MPI_Comm_free(&raft_mpi_comms);

    MPI_Finalize();
    return 0;
}
```

A RAFT developer can assume the following:
* A instance of `raft::comms::comms_t` was correctly initialized.
* All processes that are part of `raft::comms::comms_t` call into the RAFT algorithm cooperatively.

The initialized instance of `raft::comms::comms_t` can be accessed from the `raft::resources` instance:

```cpp
#include <raft/core/resources.hpp>
#include <raft/core/resource/comms.hpp>
void foo(const raft::resources& res, ...)
{
    const raft::comms_t& communicator = get_comms(res);
    const int rank = communicator.get_rank();
    const int size = communicator.get_size();
    ...
}
```
