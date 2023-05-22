# Using The Pre-Compiled Binary

At its core, RAFT is a header-only template library, which makes it very powerful in that APIs can be called with various different combinations of data types and only the templates which are actually used will be compiled into your binaries. This increased flexibility comes with a drawback that all the APIs need to be declared inline and thus calls which are made frequently in your code could be compiled again in each source file for which they are invoked.

For most functions, compile-time overhead is minimal but some of RAFT's APIs take a substantial time to compile. As a rule of thumb, most functionality in `raft::distance`, `raft::neighbors`, and `raft::cluster` is expensive to compile and most functionality in other namespaces has little compile-time overhead.

There are three ways to speed up compile times:

1. Continue to use RAFT as a header-only library and create a CUDA source file
   in your project to explicitly instantiate the templates which are slow to
   compile. This can be tedious and will still require compiling the slow code
   at least once, but it's the most flexible option if you are using types that
   aren't already compiled into `libraft`

2. If you are able to use one of the template types that are already being
   compiled into `libraft`, you can use the pre-compiled template
   instantiations, which are described in more detail in the following section.

3. If you would like to use RAFT but either cannot or would prefer not to
   compile any CUDA code yourself, you can simply add `libraft` to your link
   libraries and use the growing set of `raft::runtime` APIs.

### How do I verify template instantiations didn't compile into my binary?

To verify that you are not accidentally instantiating templates that have not been pre-compiled in RAFT, set the `RAFT_EXPLICIT_INSTANTIATE_ONLY` macro. This only works if you are linking with the pre-compiled libraft (i.e., when `RAFT_COMPILED` has been defined). To check if, for instance, `raft::distance::distance` has been precompiled with specific template arguments, you can set `RAFT_EXPLICIT_INSTANTIATE_ONLY` at the top of the file you are compiling, as in the following example:

```c++

#ifdef RAFT_COMPILED
#define RAFT_EXPLICIT_INSTANTIATE_ONLY
#endif

#include <cstdint>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>

int main()
{
  raft::resources handle{};

  // Change IdxT to uint64_t and you will get an error because you are
  // instantiating a template that has not been pre-compiled.
  using IdxT = int;

  const float* x = nullptr;
  const float* y = nullptr;
  float* out     = nullptr;
  int m          = 1024;
  int n          = 1024;
  int k          = 1024;
  bool row_major = true;
  raft::distance::distance<raft::distance::DistanceType::L1, float, float, float, IdxT>(
    handle, x, y, out, m, n, k, row_major, 2.0f);
}
```

## Runtime APIs

RAFT contains a growing list of runtime APIs that, unlike the pre-compiled
template instantiations, allow you to link against `libraft` and invoke RAFT
directly from `cpp` files. The benefit to RAFT's runtime APIs is that they can
be used from code that is compiled with a `c++` compiler (rather than the CUDA
compiler `nvcc`). This enables the `runtime` APIs to power `pylibraft`.

