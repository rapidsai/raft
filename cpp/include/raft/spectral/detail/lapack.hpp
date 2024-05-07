/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#pragma once
#include <raft/core/error.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>

#include <cusolverDn.h>

// for now; TODO: check if/where this `define` should be;
//
#define USE_LAPACK

namespace raft {

#define lapackCheckError(status)                                                     \
  {                                                                                  \
    if (status < 0) {                                                                \
      std::stringstream ss;                                                          \
      ss << "Lapack error: argument number " << -status << " had an illegal value."; \
      throw exception(ss.str());                                                     \
    } else if (status > 0)                                                           \
      RAFT_FAIL("Lapack error: internal error.");                                    \
  }

extern "C" void sgeqrf_(
  int* m, int* n, float* a, int* lda, float* tau, float* work, int* lwork, int* info);
extern "C" void dgeqrf_(
  int* m, int* n, double* a, int* lda, double* tau, double* work, int* lwork, int* info);
extern "C" void sormqr_(char* side,
                        char* trans,
                        int* m,
                        int* n,
                        int* k,
                        float* a,
                        int* lda,
                        const float* tau,
                        float* c,
                        int* ldc,
                        float* work,
                        int* lwork,
                        int* info);
extern "C" void dormqr_(char* side,
                        char* trans,
                        int* m,
                        int* n,
                        int* k,
                        double* a,
                        int* lda,
                        const double* tau,
                        double* c,
                        int* ldc,
                        double* work,
                        int* lwork,
                        int* info);
extern "C" int dgeev_(char* jobvl,
                      char* jobvr,
                      int* n,
                      double* a,
                      int* lda,
                      double* wr,
                      double* wi,
                      double* vl,
                      int* ldvl,
                      double* vr,
                      int* ldvr,
                      double* work,
                      int* lwork,
                      int* info);

extern "C" int sgeev_(char* jobvl,
                      char* jobvr,
                      int* n,
                      float* a,
                      int* lda,
                      float* wr,
                      float* wi,
                      float* vl,
                      int* ldvl,
                      float* vr,
                      int* ldvr,
                      float* work,
                      int* lwork,
                      int* info);

extern "C" cusolverStatus_t cusolverDnSgemmHost(cublasOperation_t transa,
                                                cublasOperation_t transb,
                                                int m,
                                                int n,
                                                int k,
                                                const float* alpha,
                                                const float* A,
                                                int lda,
                                                const float* B,
                                                int ldb,
                                                const float* beta,
                                                float* C,
                                                int ldc);

extern "C" cusolverStatus_t cusolverDnDgemmHost(cublasOperation_t transa,
                                                cublasOperation_t transb,
                                                int m,
                                                int n,
                                                int k,
                                                const double* alpha,
                                                const double* A,
                                                int lda,
                                                const double* B,
                                                int ldb,
                                                const double* beta,
                                                double* C,
                                                int ldc);

extern "C" cusolverStatus_t cusolverDnSsterfHost(int n, float* d, float* e, int* info);

extern "C" cusolverStatus_t cusolverDnDsterfHost(int n, double* d, double* e, int* info);

extern "C" cusolverStatus_t cusolverDnSsteqrHost(
  const signed char* compz, int n, float* d, float* e, float* z, int ldz, float* work, int* info);

extern "C" cusolverStatus_t cusolverDnDsteqrHost(const signed char* compz,
                                                 int n,
                                                 double* d,
                                                 double* e,
                                                 double* z,
                                                 int ldz,
                                                 double* work,
                                                 int* info);

template <typename T>
class Lapack {
 private:
  Lapack();
  ~Lapack();

 public:
  static void check_lapack_enabled();

  static void gemm(bool transa,
                   bool transb,
                   int m,
                   int n,
                   int k,
                   T alpha,
                   const T* A,
                   int lda,
                   const T* B,
                   int ldb,
                   T beta,
                   T* C,
                   int ldc);

  // special QR for lanczos
  static void sterf(int n, T* d, T* e);
  static void steqr(char compz, int n, T* d, T* e, T* z, int ldz, T* work);

  // QR
  // computes the QR factorization of a general matrix
  static void geqrf(int m, int n, T* a, int lda, T* tau, T* work, int* lwork);
  // Generates the real orthogonal matrix Q of the QR factorization formed by geqrf.

  // multiply C by implicit Q
  static void ormqr(bool right_side,
                    bool transq,
                    int m,
                    int n,
                    int k,
                    T* a,
                    int lda,
                    T* tau,
                    T* c,
                    int ldc,
                    T* work,
                    int* lwork);

  static void geev(T* A, T* eigenvalues, int dim, int lda);
  static void geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr);
  static void geev(T* A,
                   T* eigenvalues_r,
                   T* eigenvalues_i,
                   T* eigenvectors_r,
                   T* eigenvectors_i,
                   int dim,
                   int lda,
                   int ldvr);

 private:
  static void lapack_gemm(const char transa,
                          const char transb,
                          int m,
                          int n,
                          int k,
                          float alpha,
                          const float* a,
                          int lda,
                          const float* b,
                          int ldb,
                          float beta,
                          float* c,
                          int ldc)
  {
    cublasOperation_t cublas_transa = (transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cublas_transb = (transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cusolverDnSgemmHost(
      cublas_transa, cublas_transb, m, n, k, &alpha, (float*)a, lda, (float*)b, ldb, &beta, c, ldc);
  }

  static void lapack_gemm(const signed char transa,
                          const signed char transb,
                          int m,
                          int n,
                          int k,
                          double alpha,
                          const double* a,
                          int lda,
                          const double* b,
                          int ldb,
                          double beta,
                          double* c,
                          int ldc)
  {
    cublasOperation_t cublas_transa = (transa == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cublas_transb = (transb == 'N') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cusolverDnDgemmHost(cublas_transa,
                        cublas_transb,
                        m,
                        n,
                        k,
                        &alpha,
                        (double*)a,
                        lda,
                        (double*)b,
                        ldb,
                        &beta,
                        c,
                        ldc);
  }

  static void lapack_sterf(int n, float* d, float* e, int* info)
  {
    cusolverDnSsterfHost(n, d, e, info);
  }

  static void lapack_sterf(int n, double* d, double* e, int* info)
  {
    cusolverDnDsterfHost(n, d, e, info);
  }

  static void lapack_steqr(
    const signed char compz, int n, float* d, float* e, float* z, int ldz, float* work, int* info)
  {
    cusolverDnSsteqrHost(&compz, n, d, e, z, ldz, work, info);
  }

  static void lapack_steqr(const signed char compz,
                           int n,
                           double* d,
                           double* e,
                           double* z,
                           int ldz,
                           double* work,
                           int* info)
  {
    cusolverDnDsteqrHost(&compz, n, d, e, z, ldz, work, info);
  }

  static void lapack_geqrf(
    int m, int n, float* a, int lda, float* tau, float* work, int* lwork, int* info)
  {
    sgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
  }

  static void lapack_geqrf(
    int m, int n, double* a, int lda, double* tau, double* work, int* lwork, int* info)
  {
    dgeqrf_(&m, &n, a, &lda, tau, work, lwork, info);
  }

  static void lapack_ormqr(char side,
                           char trans,
                           int m,
                           int n,
                           int k,
                           float* a,
                           int lda,
                           float* tau,
                           float* c,
                           int ldc,
                           float* work,
                           int* lwork,
                           int* info)
  {
    sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
  }

  static void lapack_ormqr(char side,
                           char trans,
                           int m,
                           int n,
                           int k,
                           double* a,
                           int lda,
                           double* tau,
                           double* c,
                           int ldc,
                           double* work,
                           int* lwork,
                           int* info)
  {
    dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, lwork, info);
  }

  static int lapack_geev_dispatch(char* jobvl,
                                  char* jobvr,
                                  int* n,
                                  double* a,
                                  int* lda,
                                  double* wr,
                                  double* wi,
                                  double* vl,
                                  int* ldvl,
                                  double* vr,
                                  int* ldvr,
                                  double* work,
                                  int* lwork,
                                  int* info)
  {
    return dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
  }

  static int lapack_geev_dispatch(char* jobvl,
                                  char* jobvr,
                                  int* n,
                                  float* a,
                                  int* lda,
                                  float* wr,
                                  float* wi,
                                  float* vl,
                                  int* ldvl,
                                  float* vr,
                                  int* ldvr,
                                  float* work,
                                  int* lwork,
                                  int* info)
  {
    return sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
  }

  // real eigenvalues
  static void lapack_geev(T* A, T* eigenvalues, int dim, int lda)
  {
    char job = 'N';
    std::vector<T> WI(dim);
    int ldv       = 1;
    T* vl         = 0;
    int work_size = 6 * dim;
    std::vector<T> work(work_size);
    int info;
    lapack_geev_dispatch(&job,
                         &job,
                         &dim,
                         A,
                         &lda,
                         eigenvalues,
                         WI.data(),
                         vl,
                         &ldv,
                         vl,
                         &ldv,
                         work.data(),
                         &work_size,
                         &info);
    lapackCheckError(info);
  }

  // real eigenpairs
  static void lapack_geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr)
  {
    char jobvl = 'N';
    char jobvr = 'V';
    std::vector<T> WI(dim);
    int work_size = 6 * dim;
    T* vl         = 0;
    int ldvl      = 1;
    std::vector<T> work(work_size);
    int info;
    lapack_geev_dispatch(&jobvl,
                         &jobvr,
                         &dim,
                         A,
                         &lda,
                         eigenvalues,
                         WI.data(),
                         vl,
                         &ldvl,
                         eigenvectors,
                         &ldvr,
                         work.data(),
                         &work_size,
                         &info);
    lapackCheckError(info);
  }

  // complex eigenpairs
  static void lapack_geev(T* A,
                          T* eigenvalues_r,
                          T* eigenvalues_i,
                          T* eigenvectors_r,
                          T* eigenvectors_i,
                          int dim,
                          int lda,
                          int ldvr)
  {
    char jobvl    = 'N';
    char jobvr    = 'V';
    int work_size = 8 * dim;
    int ldvl      = 1;
    std::vector<T> work(work_size);
    int info;
    lapack_geev_dispatch(&jobvl,
                         &jobvr,
                         &dim,
                         A,
                         &lda,
                         eigenvalues_r,
                         eigenvalues_i,
                         0,
                         &ldvl,
                         eigenvectors_r,
                         &ldvr,
                         work.data(),
                         &work_size,
                         &info);
    lapackCheckError(info);
  }
};

template <typename T>
void Lapack<T>::check_lapack_enabled()
{
#ifndef USE_LAPACK
  RAFT_FAIL("Error: LAPACK not enabled.");
#endif
}

template <typename T>
void Lapack<T>::gemm(bool transa,
                     bool transb,
                     int m,
                     int n,
                     int k,
                     T alpha,
                     const T* A,
                     int lda,
                     const T* B,
                     int ldb,
                     T beta,
                     T* C,
                     int ldc)
{
  // check_lapack_enabled();
  // #ifdef NVGRAPH_USE_LAPACK
  const char transA_char = transa ? 'T' : 'N';
  const char transB_char = transb ? 'T' : 'N';
  lapack_gemm(transA_char, transB_char, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  // #endif
}

template <typename T>
void Lapack<T>::sterf(int n, T* d, T* e)
{
  //    check_lapack_enabled();
  // #ifdef NVGRAPH_USE_LAPACK
  int info;
  lapack_sterf(n, d, e, &info);
  lapackCheckError(info);
  // #endif
}

template <typename T>
void Lapack<T>::steqr(char compz, int n, T* d, T* e, T* z, int ldz, T* work)
{
  //    check_lapack_enabled();
  // #ifdef NVGRAPH_USE_LAPACK
  int info;
  lapack_steqr(compz, n, d, e, z, ldz, work, &info);
  lapackCheckError(info);
  // #endif
}

template <typename T>
void Lapack<T>::geqrf(int m, int n, T* a, int lda, T* tau, T* work, int* lwork)
{
  check_lapack_enabled();
#ifdef USE_LAPACK
  int info;
  lapack_geqrf(m, n, a, lda, tau, work, lwork, &info);
  lapackCheckError(info);
#endif
}
template <typename T>
void Lapack<T>::ormqr(bool right_side,
                      bool transq,
                      int m,
                      int n,
                      int k,
                      T* a,
                      int lda,
                      T* tau,
                      T* c,
                      int ldc,
                      T* work,
                      int* lwork)
{
  check_lapack_enabled();
#ifdef USE_LAPACK
  char side  = right_side ? 'R' : 'L';
  char trans = transq ? 'T' : 'N';
  int info;
  lapack_ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, &info);
  lapackCheckError(info);
#endif
}

// real eigenvalues
template <typename T>
void Lapack<T>::geev(T* A, T* eigenvalues, int dim, int lda)
{
  check_lapack_enabled();
#ifdef USE_LAPACK
  lapack_geev(A, eigenvalues, dim, lda);
#endif
}
// real eigenpairs
template <typename T>
void Lapack<T>::geev(T* A, T* eigenvalues, T* eigenvectors, int dim, int lda, int ldvr)
{
  check_lapack_enabled();
#ifdef USE_LAPACK
  lapack_geev(A, eigenvalues, eigenvectors, dim, lda, ldvr);
#endif
}
// complex eigenpairs
template <typename T>
void Lapack<T>::geev(T* A,
                     T* eigenvalues_r,
                     T* eigenvalues_i,
                     T* eigenvectors_r,
                     T* eigenvectors_i,
                     int dim,
                     int lda,
                     int ldvr)
{
  check_lapack_enabled();
#ifdef USE_LAPACK
  lapack_geev(A, eigenvalues_r, eigenvalues_i, eigenvectors_r, eigenvectors_i, dim, lda, ldvr);
#endif
}

}  // namespace raft
