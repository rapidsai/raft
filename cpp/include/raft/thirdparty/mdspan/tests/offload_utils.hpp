
namespace {
bool dispatch_host = true;

#define __MDSPAN_DEVICE_ASSERT_EQ(LHS, RHS) \
if (!(LHS == RHS)) { \
  printf("expected equality of %s and %s\n", #LHS, #RHS); \
  errors[0]++; \
}

#ifdef _MDSPAN_HAS_CUDA

template<class LAMBDA>
RAFT_KERNEL dispatch_kernel(const LAMBDA f) {
  f();
}

template<class LAMBDA>
void dispatch(LAMBDA&& f) {
  if(dispatch_host) {
    static_cast<LAMBDA&&>(f)();
  } else {
    dispatch_kernel<<<1,1>>>(static_cast<LAMBDA&&>(f));
    cudaDeviceSynchronize();
  }
}

template<class T>
T* allocate_array(size_t size) {
  T* ptr = nullptr;
  if(dispatch_host == true)
    ptr = new T[size];
  else
    cudaMallocManaged(&ptr, sizeof(T)*size);
  return ptr;
}

template<class T>
void free_array(T* ptr) {
  if(dispatch_host == true)
    delete [] ptr;
  else
    cudaFree(ptr);
}

#define __MDSPAN_TESTS_RUN_TEST(A) \
 dispatch_host = true; \
 A; \
 dispatch_host = false; \
 A;

#define __MDSPAN_TESTS_DISPATCH_DEFINED
#endif // _MDSPAN_HAS_CUDA

#ifndef __MDSPAN_TESTS_DISPATCH_DEFINED
template<class LAMBDA>
void dispatch(LAMBDA&& f) {
  static_cast<LAMBDA&&>(f)();
}
template<class T>
T* allocate_array(size_t size) {
  T* ptr = nullptr;
  ptr = new T[size];
  return ptr;
}

template<class T>
void free_array(T* ptr) {
  delete [] ptr;
}

#define __MDSPAN_TESTS_RUN_TEST(A) \
 dispatch_host = true; \
 A;
#endif
} // namespace
