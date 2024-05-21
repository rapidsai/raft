// Copyright (c) 2022-2024, NVIDIA CORPORATION.
#ifndef __CLANG_CUDA_ADDITIONAL_INTRINSICS_H__
#define __CLANG_CUDA_ADDITIONAL_INTRINSICS_H__
#ifndef __CUDA__
#error "This file is for CUDA compilation only."
#endif

// for some of these macros, see cuda_fp16.hpp
#if defined(__cplusplus) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320))
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LDG_PTR "l"
#define __LBITS   "64"
#else
#define __LDG_PTR "r"
#define __LBITS   "32"
#endif  // (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)

#define __NOARG

#define __MAKE_LD(cop, c_typ, int_typ, ptx_typ, inl_typ, mem)                          \
  __device__ __forceinline__ c_typ __ld##cop(const c_typ* addr)                        \
  {                                                                                    \
    int_typ out;                                                                       \
    asm("ld." #cop "." ptx_typ " %0, [%1];" : "=" inl_typ(out) : __LDG_PTR(addr) mem); \
    return (c_typ)out;                                                                 \
  }

#define __MAKE_LD2(cop, c_typ, int_typ, ptx_typ, inl_typ, mem)  \
  __device__ __forceinline__ c_typ __ld##cop(const c_typ* addr) \
  {                                                             \
    int_typ out1, out2;                                         \
    asm("ld." #cop ".v2." ptx_typ " {%0, %1}, [%2];"            \
        : "=" inl_typ(out1), "=" inl_typ(out2)                  \
        : __LDG_PTR(addr) mem);                                 \
    c_typ out;                                                  \
    out.x = out1;                                               \
    out.y = out2;                                               \
    return out;                                                 \
  }

#define __MAKE_LD4(cop, c_typ, int_typ, ptx_typ, inl_typ, mem)                       \
  __device__ __forceinline__ c_typ __ld##cop(const c_typ* addr)                      \
  {                                                                                  \
    int_typ out1, out2, out3, out4;                                                  \
    asm("ld." #cop ".v4." ptx_typ " {%0, %1, %2, %3}, [%4];"                         \
        : "=" inl_typ(out1), "=" inl_typ(out2), "=" inl_typ(out3), "=" inl_typ(out4) \
        : __LDG_PTR(addr) mem);                                                      \
    c_typ out;                                                                       \
    out.x = out1;                                                                    \
    out.y = out2;                                                                    \
    out.z = out3;                                                                    \
    out.w = out4;                                                                    \
    return out;                                                                      \
  }

__MAKE_LD(cg, char, short, "s8", "h", __NOARG)
__MAKE_LD(cg, signed char, short, "s8", "h", __NOARG)
__MAKE_LD(cg, unsigned char, short, "u8", "h", __NOARG)
__MAKE_LD(cg, short, short, "s16", "h", __NOARG)
__MAKE_LD(cg, unsigned short, unsigned short, "u16", "h", __NOARG)
__MAKE_LD(cg, int, int, "s32", "r", __NOARG)
__MAKE_LD(cg, unsigned int, unsigned int, "u32", "r", __NOARG)
__MAKE_LD(cg, long, long, "s" __LBITS, __LDG_PTR, __NOARG)
__MAKE_LD(cg, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR, __NOARG)
__MAKE_LD(cg, long long, long long, "s64", "l", __NOARG)
__MAKE_LD(cg, unsigned long long, unsigned long long, "u64", "l", __NOARG)
__MAKE_LD(cg, float, float, "f32", "f", __NOARG)
__MAKE_LD(cg, double, double, "f64", "d", __NOARG)

__MAKE_LD2(cg, char2, short, "s8", "h", __NOARG)
__MAKE_LD2(cg, uchar2, short, "u8", "h", __NOARG)
__MAKE_LD2(cg, short2, short, "s16", "h", __NOARG)
__MAKE_LD2(cg, ushort2, unsigned short, "u16", "h", __NOARG)
__MAKE_LD2(cg, int2, int, "s32", "r", __NOARG)
__MAKE_LD2(cg, uint2, unsigned int, "u32", "r", __NOARG)
__MAKE_LD2(cg, longlong2, long long, "s64", "l", __NOARG)
__MAKE_LD2(cg, ulonglong2, unsigned long long, "u64", "l", __NOARG)
__MAKE_LD2(cg, float2, float, "f32", "f", __NOARG)
__MAKE_LD2(cg, double2, double, "f64", "d", __NOARG)

__MAKE_LD4(cg, char4, short, "s8", "h", __NOARG)
__MAKE_LD4(cg, uchar4, short, "u8", "h", __NOARG)
__MAKE_LD4(cg, short4, short, "s16", "h", __NOARG)
__MAKE_LD4(cg, ushort4, unsigned short, "u16", "h", __NOARG)
__MAKE_LD4(cg, int4, int, "s32", "r", __NOARG)
__MAKE_LD4(cg, uint4, unsigned int, "u32", "r", __NOARG)
__MAKE_LD4(cg, float4, float, "f32", "f", __NOARG)

__MAKE_LD(ca, char, short, "s8", "h", __NOARG)
__MAKE_LD(ca, signed char, short, "s8", "h", __NOARG)
__MAKE_LD(ca, unsigned char, short, "u8", "h", __NOARG)
__MAKE_LD(ca, short, short, "s16", "h", __NOARG)
__MAKE_LD(ca, unsigned short, unsigned short, "u16", "h", __NOARG)
__MAKE_LD(ca, int, int, "s32", "r", __NOARG)
__MAKE_LD(ca, unsigned int, unsigned int, "u32", "r", __NOARG)
__MAKE_LD(ca, long, long, "s" __LBITS, __LDG_PTR, __NOARG)
__MAKE_LD(ca, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR, __NOARG)
__MAKE_LD(ca, long long, long long, "s64", "l", __NOARG)
__MAKE_LD(ca, unsigned long long, unsigned long long, "u64", "l", __NOARG)
__MAKE_LD(ca, float, float, "f32", "f", __NOARG)
__MAKE_LD(ca, double, double, "f64", "d", __NOARG)

__MAKE_LD2(ca, char2, short, "s8", "h", __NOARG)
__MAKE_LD2(ca, uchar2, short, "u8", "h", __NOARG)
__MAKE_LD2(ca, short2, short, "s16", "h", __NOARG)
__MAKE_LD2(ca, ushort2, unsigned short, "u16", "h", __NOARG)
__MAKE_LD2(ca, int2, int, "s32", "r", __NOARG)
__MAKE_LD2(ca, uint2, unsigned int, "u32", "r", __NOARG)
__MAKE_LD2(ca, longlong2, long long, "s64", "l", __NOARG)
__MAKE_LD2(ca, ulonglong2, unsigned long long, "u64", "l", __NOARG)
__MAKE_LD2(ca, float2, float, "f32", "f", __NOARG)
__MAKE_LD2(ca, double2, double, "f64", "d", __NOARG)

__MAKE_LD4(ca, char4, short, "s8", "h", __NOARG)
__MAKE_LD4(ca, uchar4, short, "u8", "h", __NOARG)
__MAKE_LD4(ca, short4, short, "s16", "h", __NOARG)
__MAKE_LD4(ca, ushort4, unsigned short, "u16", "h", __NOARG)
__MAKE_LD4(ca, int4, int, "s32", "r", __NOARG)
__MAKE_LD4(ca, uint4, unsigned int, "u32", "r", __NOARG)
__MAKE_LD4(ca, float4, float, "f32", "f", __NOARG)

__MAKE_LD(cs, char, short, "s8", "h", __NOARG)
__MAKE_LD(cs, signed char, short, "s8", "h", __NOARG)
__MAKE_LD(cs, unsigned char, short, "u8", "h", __NOARG)
__MAKE_LD(cs, short, short, "s16", "h", __NOARG)
__MAKE_LD(cs, unsigned short, unsigned short, "u16", "h", __NOARG)
__MAKE_LD(cs, int, int, "s32", "r", __NOARG)
__MAKE_LD(cs, unsigned int, unsigned int, "u32", "r", __NOARG)
__MAKE_LD(cs, long, long, "s" __LBITS, __LDG_PTR, __NOARG)
__MAKE_LD(cs, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR, __NOARG)
__MAKE_LD(cs, long long, long long, "s64", "l", __NOARG)
__MAKE_LD(cs, unsigned long long, unsigned long long, "u64", "l", __NOARG)
__MAKE_LD(cs, float, float, "f32", "f", __NOARG)
__MAKE_LD(cs, double, double, "f64", "d", __NOARG)

__MAKE_LD2(cs, char2, short, "s8", "h", __NOARG)
__MAKE_LD2(cs, uchar2, short, "u8", "h", __NOARG)
__MAKE_LD2(cs, short2, short, "s16", "h", __NOARG)
__MAKE_LD2(cs, ushort2, unsigned short, "u16", "h", __NOARG)
__MAKE_LD2(cs, int2, int, "s32", "r", __NOARG)
__MAKE_LD2(cs, uint2, unsigned int, "u32", "r", __NOARG)
__MAKE_LD2(cs, longlong2, long long, "s64", "l", __NOARG)
__MAKE_LD2(cs, ulonglong2, unsigned long long, "u64", "l", __NOARG)
__MAKE_LD2(cs, float2, float, "f32", "f", __NOARG)
__MAKE_LD2(cs, double2, double, "f64", "d", __NOARG)

__MAKE_LD4(cs, char4, short, "s8", "h", __NOARG)
__MAKE_LD4(cs, uchar4, short, "u8", "h", __NOARG)
__MAKE_LD4(cs, short4, short, "s16", "h", __NOARG)
__MAKE_LD4(cs, ushort4, unsigned short, "u16", "h", __NOARG)
__MAKE_LD4(cs, int4, int, "s32", "r", __NOARG)
__MAKE_LD4(cs, uint4, unsigned int, "u32", "r", __NOARG)
__MAKE_LD4(cs, float4, float, "f32", "f", __NOARG)

__MAKE_LD(lu, char, short, "s8", "h", : "memory")
__MAKE_LD(lu, signed char, short, "s8", "h", : "memory")
__MAKE_LD(lu, unsigned char, short, "u8", "h", : "memory")
__MAKE_LD(lu, short, short, "s16", "h", : "memory")
__MAKE_LD(lu, unsigned short, unsigned short, "u16", "h", : "memory")
__MAKE_LD(lu, int, int, "s32", "r", : "memory")
__MAKE_LD(lu, unsigned int, unsigned int, "u32", "r", : "memory")
__MAKE_LD(lu, long, long, "s" __LBITS, __LDG_PTR, : "memory")
__MAKE_LD(lu, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR, : "memory")
__MAKE_LD(lu, long long, long long, "s64", "l", : "memory")
__MAKE_LD(lu, unsigned long long, unsigned long long, "u64", "l", : "memory")
__MAKE_LD(lu, float, float, "f32", "f", : "memory")
__MAKE_LD(lu, double, double, "f64", "d", : "memory")

__MAKE_LD2(lu, char2, short, "s8", "h", : "memory")
__MAKE_LD2(lu, uchar2, short, "u8", "h", : "memory")
__MAKE_LD2(lu, short2, short, "s16", "h", : "memory")
__MAKE_LD2(lu, ushort2, unsigned short, "u16", "h", : "memory")
__MAKE_LD2(lu, int2, int, "s32", "r", : "memory")
__MAKE_LD2(lu, uint2, unsigned int, "u32", "r", : "memory")
__MAKE_LD2(lu, longlong2, long long, "s64", "l", : "memory")
__MAKE_LD2(lu, ulonglong2, unsigned long long, "u64", "l", : "memory")
__MAKE_LD2(lu, float2, float, "f32", "f", : "memory")
__MAKE_LD2(lu, double2, double, "f64", "d", : "memory")

__MAKE_LD4(lu, char4, short, "s8", "h", : "memory")
__MAKE_LD4(lu, uchar4, short, "u8", "h", : "memory")
__MAKE_LD4(lu, short4, short, "s16", "h", : "memory")
__MAKE_LD4(lu, ushort4, unsigned short, "u16", "h", : "memory")
__MAKE_LD4(lu, int4, int, "s32", "r", : "memory")
__MAKE_LD4(lu, uint4, unsigned int, "u32", "r", : "memory")
__MAKE_LD4(lu, float4, float, "f32", "f", : "memory")

__MAKE_LD(cv, char, short, "s8", "h", : "memory")
__MAKE_LD(cv, signed char, short, "s8", "h", : "memory")
__MAKE_LD(cv, unsigned char, short, "u8", "h", : "memory")
__MAKE_LD(cv, short, short, "s16", "h", : "memory")
__MAKE_LD(cv, unsigned short, unsigned short, "u16", "h", : "memory")
__MAKE_LD(cv, int, int, "s32", "r", : "memory")
__MAKE_LD(cv, unsigned int, unsigned int, "u32", "r", : "memory")
__MAKE_LD(cv, long, long, "s" __LBITS, __LDG_PTR, : "memory")
__MAKE_LD(cv, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR, : "memory")
__MAKE_LD(cv, long long, long long, "s64", "l", : "memory")
__MAKE_LD(cv, unsigned long long, unsigned long long, "u64", "l", : "memory")
__MAKE_LD(cv, float, float, "f32", "f", : "memory")
__MAKE_LD(cv, double, double, "f64", "d", : "memory")

__MAKE_LD2(cv, char2, short, "s8", "h", : "memory")
__MAKE_LD2(cv, uchar2, short, "u8", "h", : "memory")
__MAKE_LD2(cv, short2, short, "s16", "h", : "memory")
__MAKE_LD2(cv, ushort2, unsigned short, "u16", "h", : "memory")
__MAKE_LD2(cv, int2, int, "s32", "r", : "memory")
__MAKE_LD2(cv, uint2, unsigned int, "u32", "r", : "memory")
__MAKE_LD2(cv, longlong2, long long, "s64", "l", : "memory")
__MAKE_LD2(cv, ulonglong2, unsigned long long, "u64", "l", : "memory")
__MAKE_LD2(cv, float2, float, "f32", "f", : "memory")
__MAKE_LD2(cv, double2, double, "f64", "d", : "memory")

__MAKE_LD4(cv, char4, short, "s8", "h", : "memory")
__MAKE_LD4(cv, uchar4, short, "u8", "h", : "memory")
__MAKE_LD4(cv, short4, short, "s16", "h", : "memory")
__MAKE_LD4(cv, ushort4, unsigned short, "u16", "h", : "memory")
__MAKE_LD4(cv, int4, int, "s32", "r", : "memory")
__MAKE_LD4(cv, uint4, unsigned int, "u32", "r", : "memory")
__MAKE_LD4(cv, float4, float, "f32", "f", : "memory")

#define __MAKE_ST(cop, c_typ, int_typ, ptx_typ, inl_typ)                                        \
  __device__ __forceinline__ void __st##cop(c_typ* addr, c_typ v)                               \
  {                                                                                             \
    asm("st." #cop "." ptx_typ " [%0], %1;" ::__LDG_PTR(addr), inl_typ((int_typ)v) : "memory"); \
  }

#define __MAKE_ST2(cop, c_typ, int_typ, ptx_typ, inl_typ)                                        \
  __device__ __forceinline__ void __st##cop(c_typ* addr, c_typ v)                                \
  {                                                                                              \
    int_typ v1 = v.x, v2 = v.y;                                                                  \
    asm("st." #cop ".v2." ptx_typ " [%0], {%1, %2};" ::__LDG_PTR(addr), inl_typ(v1), inl_typ(v2) \
        : "memory");                                                                             \
  }

#define __MAKE_ST4(cop, c_typ, int_typ, ptx_typ, inl_typ)                       \
  __device__ __forceinline__ void __st##cop(c_typ* addr, c_typ v)               \
  {                                                                             \
    int_typ v1 = v.x, v2 = v.y, v3 = v.z, v4 = v.w;                             \
    asm("st." #cop ".v4." ptx_typ " [%0], {%1, %2, %3, %4};" ::__LDG_PTR(addr), \
        inl_typ(v1),                                                            \
        inl_typ(v2),                                                            \
        inl_typ(v3),                                                            \
        inl_typ(v4)                                                             \
        : "memory");                                                            \
  }

__MAKE_ST(wb, char, short, "s8", "h")
__MAKE_ST(wb, signed char, short, "s8", "h")
__MAKE_ST(wb, unsigned char, short, "u8", "h")
__MAKE_ST(wb, short, short, "s16", "h")
__MAKE_ST(wb, unsigned short, unsigned short, "u16", "h")
__MAKE_ST(wb, int, int, "s32", "r")
__MAKE_ST(wb, unsigned int, unsigned int, "u32", "r")
__MAKE_ST(wb, long, long, "s" __LBITS, __LDG_PTR)
__MAKE_ST(wb, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR)
__MAKE_ST(wb, long long, long long, "s64", "l")
__MAKE_ST(wb, unsigned long long, unsigned long long, "u64", "l")
__MAKE_ST(wb, float, float, "f32", "f")
__MAKE_ST(wb, double, double, "f64", "d")

__MAKE_ST2(wb, char2, short, "s8", "h")
__MAKE_ST2(wb, uchar2, short, "u8", "h")
__MAKE_ST2(wb, short2, short, "s16", "h")
__MAKE_ST2(wb, ushort2, unsigned short, "u16", "h")
__MAKE_ST2(wb, int2, int, "s32", "r")
__MAKE_ST2(wb, uint2, unsigned int, "u32", "r")
__MAKE_ST2(wb, longlong2, long long, "s64", "l")
__MAKE_ST2(wb, ulonglong2, unsigned long long, "u64", "l")
__MAKE_ST2(wb, float2, float, "f32", "f")
__MAKE_ST2(wb, double2, double, "f64", "d")

__MAKE_ST4(wb, char4, short, "s8", "h")
__MAKE_ST4(wb, uchar4, short, "u8", "h")
__MAKE_ST4(wb, short4, short, "s16", "h")
__MAKE_ST4(wb, ushort4, unsigned short, "u16", "h")
__MAKE_ST4(wb, int4, int, "s32", "r")
__MAKE_ST4(wb, uint4, unsigned int, "u32", "r")
__MAKE_ST4(wb, float4, float, "f32", "f")

__MAKE_ST(cg, char, short, "s8", "h")
__MAKE_ST(cg, signed char, short, "s8", "h")
__MAKE_ST(cg, unsigned char, short, "u8", "h")
__MAKE_ST(cg, short, short, "s16", "h")
__MAKE_ST(cg, unsigned short, unsigned short, "u16", "h")
__MAKE_ST(cg, int, int, "s32", "r")
__MAKE_ST(cg, unsigned int, unsigned int, "u32", "r")
__MAKE_ST(cg, long, long, "s" __LBITS, __LDG_PTR)
__MAKE_ST(cg, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR)
__MAKE_ST(cg, long long, long long, "s64", "l")
__MAKE_ST(cg, unsigned long long, unsigned long long, "u64", "l")
__MAKE_ST(cg, float, float, "f32", "f")
__MAKE_ST(cg, double, double, "f64", "d")

__MAKE_ST2(cg, char2, short, "s8", "h")
__MAKE_ST2(cg, uchar2, short, "u8", "h")
__MAKE_ST2(cg, short2, short, "s16", "h")
__MAKE_ST2(cg, ushort2, unsigned short, "u16", "h")
__MAKE_ST2(cg, int2, int, "s32", "r")
__MAKE_ST2(cg, uint2, unsigned int, "u32", "r")
__MAKE_ST2(cg, longlong2, long long, "s64", "l")
__MAKE_ST2(cg, ulonglong2, unsigned long long, "u64", "l")
__MAKE_ST2(cg, float2, float, "f32", "f")
__MAKE_ST2(cg, double2, double, "f64", "d")

__MAKE_ST4(cg, char4, short, "s8", "h")
__MAKE_ST4(cg, uchar4, short, "u8", "h")
__MAKE_ST4(cg, short4, short, "s16", "h")
__MAKE_ST4(cg, ushort4, unsigned short, "u16", "h")
__MAKE_ST4(cg, int4, int, "s32", "r")
__MAKE_ST4(cg, uint4, unsigned int, "u32", "r")
__MAKE_ST4(cg, float4, float, "f32", "f")

__MAKE_ST(cs, char, short, "s8", "h")
__MAKE_ST(cs, signed char, short, "s8", "h")
__MAKE_ST(cs, unsigned char, short, "u8", "h")
__MAKE_ST(cs, short, short, "s16", "h")
__MAKE_ST(cs, unsigned short, unsigned short, "u16", "h")
__MAKE_ST(cs, int, int, "s32", "r")
__MAKE_ST(cs, unsigned int, unsigned int, "u32", "r")
__MAKE_ST(cs, long, long, "s" __LBITS, __LDG_PTR)
__MAKE_ST(cs, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR)
__MAKE_ST(cs, long long, long long, "s64", "l")
__MAKE_ST(cs, unsigned long long, unsigned long long, "u64", "l")
__MAKE_ST(cs, float, float, "f32", "f")
__MAKE_ST(cs, double, double, "f64", "d")

__MAKE_ST2(cs, char2, short, "s8", "h")
__MAKE_ST2(cs, uchar2, short, "u8", "h")
__MAKE_ST2(cs, short2, short, "s16", "h")
__MAKE_ST2(cs, ushort2, unsigned short, "u16", "h")
__MAKE_ST2(cs, int2, int, "s32", "r")
__MAKE_ST2(cs, uint2, unsigned int, "u32", "r")
__MAKE_ST2(cs, longlong2, long long, "s64", "l")
__MAKE_ST2(cs, ulonglong2, unsigned long long, "u64", "l")
__MAKE_ST2(cs, float2, float, "f32", "f")
__MAKE_ST2(cs, double2, double, "f64", "d")

__MAKE_ST4(cs, char4, short, "s8", "h")
__MAKE_ST4(cs, uchar4, short, "u8", "h")
__MAKE_ST4(cs, short4, short, "s16", "h")
__MAKE_ST4(cs, ushort4, unsigned short, "u16", "h")
__MAKE_ST4(cs, int4, int, "s32", "r")
__MAKE_ST4(cs, uint4, unsigned int, "u32", "r")
__MAKE_ST4(cs, float4, float, "f32", "f")

__MAKE_ST(wt, char, short, "s8", "h")
__MAKE_ST(wt, signed char, short, "s8", "h")
__MAKE_ST(wt, unsigned char, short, "u8", "h")
__MAKE_ST(wt, short, short, "s16", "h")
__MAKE_ST(wt, unsigned short, unsigned short, "u16", "h")
__MAKE_ST(wt, int, int, "s32", "r")
__MAKE_ST(wt, unsigned int, unsigned int, "u32", "r")
__MAKE_ST(wt, long, long, "s" __LBITS, __LDG_PTR)
__MAKE_ST(wt, unsigned long, unsigned long, "u" __LBITS, __LDG_PTR)
__MAKE_ST(wt, long long, long long, "s64", "l")
__MAKE_ST(wt, unsigned long long, unsigned long long, "u64", "l")
__MAKE_ST(wt, float, float, "f32", "f")
__MAKE_ST(wt, double, double, "f64", "d")

__MAKE_ST2(wt, char2, short, "s8", "h")
__MAKE_ST2(wt, uchar2, short, "u8", "h")
__MAKE_ST2(wt, short2, short, "s16", "h")
__MAKE_ST2(wt, ushort2, unsigned short, "u16", "h")
__MAKE_ST2(wt, int2, int, "s32", "r")
__MAKE_ST2(wt, uint2, unsigned int, "u32", "r")
__MAKE_ST2(wt, longlong2, long long, "s64", "l")
__MAKE_ST2(wt, ulonglong2, unsigned long long, "u64", "l")
__MAKE_ST2(wt, float2, float, "f32", "f")
__MAKE_ST2(wt, double2, double, "f64", "d")

__MAKE_ST4(wt, char4, short, "s8", "h")
__MAKE_ST4(wt, uchar4, short, "u8", "h")
__MAKE_ST4(wt, short4, short, "s16", "h")
__MAKE_ST4(wt, ushort4, unsigned short, "u16", "h")
__MAKE_ST4(wt, int4, int, "s32", "r")
__MAKE_ST4(wt, uint4, unsigned int, "u32", "r")
__MAKE_ST4(wt, float4, float, "f32", "f")

#undef __MAKE_ST4
#undef __MAKE_ST2
#undef __MAKE_ST
#undef __MAKE_LD4
#undef __MAKE_LD2
#undef __MAKE_LD
#undef __NOARG
#undef __LBITS
#undef __LDG_PTR

#endif  // defined(__cplusplus) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 320))

#endif  // defined(__CLANG_CUDA_ADDITIONAL_INTRINSICS_H__)
