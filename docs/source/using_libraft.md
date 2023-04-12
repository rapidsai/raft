# Using The Pre-Compiled Binary

At its core, RAFT is a header-only template library, which makes it very powerful in that APIs can be called with various different combinations of data types and only the templates which are actually used will be compiled into your binaries. This increased flexibility comes with a drawback that all the APIs need to be declared inline and thus calls which are made frequently in your code could be compiled again in each source file for which they are invoked.

For most functions, compile-time overhead is minimal but some of RAFT's APIs take a substantial time to compile. As a rule of thumb, most functionality in `raft::distance`, `raft::neighbors`, and `raft::spatial` is expensive to compile and most functionality in other namespaces has little compile-time overhead.


To speed up compilation when using RAFT as a header-only library, you can do the following... 

To speed up compilation when using the precompiled RAFT library, you can do the
following:

1. 


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
   libraries and use the growing set of runtime APIs.

## Using Template Specializations

As mentioned above, the pre-compiled template instantiations can save a lot of time if you are able to use the type combinations for the templates which are already specialized in the `libraft` binary. This will, of course, mean that you will need to add `libraft` to your link libraries.

At the top level of each namespace containing pre-compiled template specializations is a header file called `specializations.cuh`. This header file includes `extern template` directives for all the specializations which are compiled into libraft. As an example, including `raft/neighbors/specializations.cuh` in one of your source files will effectively tell the compiler to skip over any of the template specializations that are already compiled into the `libraft` binary.

### How do I verify template specializations didn't compile into my binary?

Which specializations were chosen to instantiations were based on compile time analysis and reuse. This means you can't assume that all specializations are for the public API itself. Take the following example in `raft/neighbors/specializations/detail/ivf_pq_compute_similarity.cuh`:

```c++
namespace raft::neighbors::ivf_pq::detail {

namespace {
using fp8s_t = fp_8bit<5, true>;
using fp8u_t = fp_8bit<5, false>;
}  // namespace

#define RAFT_INST(OutT, LutT)                                                                     \
  extern template auto get_compute_similarity_kernel<OutT, LutT, true, true>(uint32_t, uint32_t)  \
    ->compute_similarity_kernel_t<OutT, LutT>;                                                    \
  extern template auto get_compute_similarity_kernel<OutT, LutT, true, false>(uint32_t, uint32_t) \
    ->compute_similarity_kernel_t<OutT, LutT>;                                                    \
  extern template auto get_compute_similarity_kernel<OutT, LutT, false, true>(uint32_t, uint32_t) \
    ->compute_similarity_kernel_t<OutT, LutT>;

#define RAFT_INST_ALL_OUT_T(LutT) \
  RAFT_INST(float, LutT)          \
  RAFT_INST(half, LutT)

RAFT_INST_ALL_OUT_T(float)
RAFT_INST_ALL_OUT_T(half)
RAFT_INST_ALL_OUT_T(fp8s_t)
RAFT_INST_ALL_OUT_T(fp8u_t)

#undef RAFT_INST
#undef RAFT_INST_ALL_OUT_T

}  // namespace raft::neighbors::ivf_pq::detail
```

We can see here that the function `raft::neighbors::ivf_pq::detail::get_compute_similarity_kernel` is being instantiated for the cartesian product of `OutT={float, half, fp8s_t, fp8u_t}` and `LutT={float, half}`. After linking against the `libraft` binary and including `raft/neighbors/specializations.cuh` in your source file, you can invoke the `raft::neighbors::ivf_pq` functions and compile your code. If the specializations are working, you should be able to use `nm -g -C --defined-only /path/to/your/binary | grep raft::neighbors::ivf_pq::detail::get_compute_similarity::kernel` and you shouldn't see any results, because those symbols should be coming from the `libraft` binary and skipped from compiling into your binary.

## Runtime APIs

RAFT contains a growing list of runtime APIs that, unlike the pre-compiled template specializations, allow you to link against `libraft` and invoke RAFT directly from `cpp` files. The benefit to RAFT's runtime APIs are two-fold- unlike the template specializations, which still require your code be compiled with the CUDA compiler (`nvcc`), the `runtime` APIs are the lightweight wrappers which enable `pylibraft`.

Similar to the pre-compiled template specializations, RAFT's runtime APIs 
