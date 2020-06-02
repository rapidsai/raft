/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

// #include <cuda.h>
// #include <cublas_v2.h>
// #include <curand.h>
// #include <cusolverDn.h>
// #include <cusparse.h>

#include <rmm/thrust_rmm_allocator.h>

// CUDA block size
#define BLOCK_SIZE 1024

// Get index of matrix entry
#define IDX(i, j, lda) ((i) + (j) * (lda))

namespace raft {
namespace matrix {
  void check_size(size_t sz)
  {
    if (sz > INT_MAX) FatalError("Vector larger than INT_MAX", ERR_BAD_PARAMETERS);
  }
  template <typename ValueType_>
  void nrm1_raw_vec(ValueType_* vec, size_t n, ValueType_* res, cudaStream_t stream)
  {
    thrust::device_ptr<ValueType_> dev_ptr(vec);
    *res = thrust::reduce(dev_ptr, dev_ptr + n);
    cudaCheckError();
  }

  template <typename ValueType_>
  void fill_raw_vec(ValueType_* vec, size_t n, ValueType_ value, cudaStream_t stream)
  {
    thrust::device_ptr<ValueType_> dev_ptr(vec);
    thrust::fill(dev_ptr, dev_ptr + n, value);
    cudaCheckError();
  }

  template <typename ValueType_>
  void dump_raw_vec(ValueType_* vec, size_t n, int offset, cudaStream_t stream)
  {
#ifdef DEBUG
    thrust::device_ptr<ValueType_> dev_ptr(vec);
    COUT().precision(15);
    COUT() << "sample size = " << n << ", offset = " << offset << std::endl;
    thrust::copy(
                 dev_ptr + offset, dev_ptr + offset + n, std::ostream_iterator<ValueType_>(COUT(), " "));
    cudaCheckError();
    COUT() << std::endl;
#endif
  }

  template <typename ValueType_>
  __global__ void flag_zeroes_kernel(int num_vertices, ValueType_* vec, int* flags)
  {
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int r = tidx; r < num_vertices; r += blockDim.x * gridDim.x) {
      if (vec[r] != 0.0)
        flags[r] = 1;  // NOTE 2 : alpha*0 + (1-alpha)*1 = (1-alpha)
      else
        flags[r] = 0;
    }
  }
  template <typename ValueType_>
  __global__ void dmv0_kernel(const ValueType_* __restrict__ D,
                              const ValueType_* __restrict__ x,
                              ValueType_* __restrict__ y,
                              int n)
  {
    // y=D*x
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tidx; i < n; i += blockDim.x * gridDim.x) y[i] = D[i] * x[i];
  }
  template <typename ValueType_>
  __global__ void dmv1_kernel(const ValueType_* __restrict__ D,
                              const ValueType_* __restrict__ x,
                              ValueType_* __restrict__ y,
                              int n)
  {
    // y+=D*x
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tidx; i < n; i += blockDim.x * gridDim.x) y[i] += D[i] * x[i];
  }
  template <typename ValueType_>
  void copy_vec(ValueType_* vec1, size_t n, ValueType_* res, cudaStream_t stream)
  {
    thrust::device_ptr<ValueType_> dev_ptr(vec1);
    thrust::device_ptr<ValueType_> res_ptr(res);
#ifdef DEBUG
    // COUT() << "copy "<< n << " elements" << std::endl;
#endif
    thrust::copy_n(dev_ptr, n, res_ptr);
    cudaCheckError();
    // dump_raw_vec (res, n, 0);
  }

  template <typename ValueType_>
  void flag_zeros_raw_vec(size_t num_vertices, ValueType_* vec, int* flags, cudaStream_t stream)
  {
    int items_per_thread = 4;
    int num_threads      = 128;
    int max_grid_size    = 4096;
    check_size(num_vertices);
    int n          = static_cast<int>(num_vertices);
    int num_blocks = std::min(max_grid_size, (n / (items_per_thread * num_threads)) + 1);
    flag_zeroes_kernel<<<num_blocks, num_threads, 0, stream>>>(num_vertices, vec, flags);
    cudaCheckError();
  }

  template <typename ValueType_>
  void dmv(size_t num_vertices,
           ValueType_ alpha,
           ValueType_* D,
           ValueType_* x,
           ValueType_ beta,
           ValueType_* y,
           cudaStream_t stream)
  {
    int items_per_thread = 4;
    int num_threads      = 128;
    int max_grid_size    = 4096;
    check_size(num_vertices);
    int n          = static_cast<int>(num_vertices);
    int num_blocks = std::min(max_grid_size, (n / (items_per_thread * num_threads)) + 1);
    if (alpha == 1.0 && beta == 0.0)
      dmv0_kernel<<<num_blocks, num_threads, 0, stream>>>(D, x, y, n);
    else if (alpha == 1.0 && beta == 1.0)
      dmv1_kernel<<<num_blocks, num_threads, 0, stream>>>(D, x, y, n);
    else
      FatalError("Not implemented case of y = D*x", ERR_BAD_PARAMETERS);

    cudaCheckError();
  }

  template <typename IndexType_, typename ValueType_>
  void set_connectivity(size_t n,
                        IndexType_ root,
                        ValueType_ self_loop_val,
                        ValueType_ unreachable_val,
                        ValueType_* res,
                        cudaStream_t stream)
  {
    fill_raw_vec(res, n, unreachable_val);
    cudaMemcpy(&res[root], &self_loop_val, sizeof(self_loop_val), cudaMemcpyHostToDevice);
    cudaCheckError();
  }


  /*! A Vector contains a device vector of size |E| and type T
   */
  template <typename ValueType_>
  class Vector {
  public:
    typedef ValueType_ ValueType;

  protected:
    rmm::device_vector<ValueType> values;

  public:
    /*! Construct an empty \p Vector.
     */
    Vector(void) {}
    ~Vector(void) {}
    /*! Construct a \p Vector of size vertices.
     *
     *  \param vertices The size of the Vector
     */
    Vector(size_t vertices, cudaStream_t stream = 0)
      : values(vertices) {}
    
    size_t get_size() const { return values.size(); }
    size_t bytes() const { return values.size()*sizeof(ValueType);}
    ValueType const *raw() const { return values.data().get();  }
    ValueType *raw() { return values.data().get();  }

    void allocate(size_t n, cudaStream_t stream = 0) 
    {
      values.resize(n);
    }

    void fill(ValueType val, cudaStream_t stream = 0) 
    {
      fill_raw_vec(this->raw(), this->get_size(), val, stream); 
    } 

    void copy(Vector<ValueType> &vec1, cudaStream_t stream = 0)
    {
      if (this->get_size() == 0 && vec1.get_size()>0) {
        allocate(vec1.get_size(), stream);
        copy_vec(vec1.raw(), this->get_size(), this->raw(), stream);
      } else if (this->get_size() == vec1.get_size()) 
        copy_vec(vec1.raw(),  this->get_size(), this->raw(), stream);
      else if (this->get_size() > vec1.get_size()) {
        copy_vec(vec1.raw(),  vec1.get_size(), this->raw(), stream);
      } else {
        FatalError("Cannot copy a vector into a smaller one", ERR_BAD_PARAMETERS);
      }
    }

    ValueType nrm1(cudaStream_t stream = 0) { 
      ValueType res = 0;
      nrm1_raw_vec(this->raw(), this->get_size(), &res, stream);
      return res;
    }
  }; // class Vector

  /// Abstract matrix class
  /** Derived classes must implement matrix-vector products.
   */
  template <typename IndexType_, typename ValueType_>
  class Matrix {
  public:
    /// Number of rows
    const IndexType_ m;
    /// Number of columns
    const IndexType_ n;
    /// CUDA stream
    cudaStream_t s;  

    /// Constructor
    /** @param _m Number of rows.
     *  @param _n Number of columns.
     */
    Matrix(IndexType_ _m, IndexType_ _n) : m(_m), n(_n), s(0){}

    /// Destructor
    virtual ~Matrix() {}


    /// Get and Set CUDA stream  
    virtual void setCUDAStream(cudaStream_t _s) = 0;  
    virtual void getCUDAStream(cudaStream_t *_s) = 0;    

    /// Matrix-vector product
    /** y is overwritten with alpha*A*x+beta*y.
     *
     *  @param alpha Scalar.
     *  @param x (Input, device memory, n entries) Vector.
     *  @param beta Scalar.
     *  @param y (Input/output, device memory, m entries) Output
     *    vector.
     */
    virtual void mv(ValueType_ alpha,
		    const ValueType_ * __restrict__ x,
		    ValueType_ beta,
		    ValueType_ * __restrict__ y) const = 0;

    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const = 0;  
    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const = 0;  
    virtual void reorder(IndexType_ *p) const = 0;  

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M) = 0;
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const = 0; 
    
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const = 0;
  };

  /// Dense matrix class
  template <typename IndexType_, typename ValueType_>
  class DenseMatrix : public Matrix<IndexType_, ValueType_> {

  private:
    /// Whether to transpose matrix
    const bool trans;
    /// Matrix entries, stored column-major in device memory
    const ValueType_ * A;
    /// Leading dimension of matrix entry array
    const IndexType_ lda;

  public:
    /// Constructor
    DenseMatrix(bool _trans,
		IndexType_ _m, IndexType_ _n,
		const ValueType_ * _A, IndexType_ _lda);

    /// Destructor
    virtual ~DenseMatrix();

    /// Get and Set CUDA stream  
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);     

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
		    ValueType_ beta, ValueType_ * __restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;  

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const; 
    
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  /// Sparse matrix class in CSR format
  template <typename IndexType_, typename ValueType_>
  class CsrMatrix : public Matrix<IndexType_, ValueType_> {

  private:
    /// Whether to transpose matrix
    const bool trans;
    /// Whether matrix is stored in symmetric format
    const bool sym;
    /// Number of non-zero entries
    const IndexType_ nnz;
    /// Matrix properties
    const cusparseMatDescr_t descrA;
    /// Matrix entry values (device memory)
    /*const*/ ValueType_ * csrValA;
    /// Pointer to first entry in each row (device memory)
    const IndexType_ * csrRowPtrA;
    /// Column index of each matrix entry (device memory)
    const IndexType_ * csrColIndA;
    /// Analysis info (pointer to opaque CUSPARSE struct)  
    cusparseSolveAnalysisInfo_t info_l;
    cusparseSolveAnalysisInfo_t info_u;  
    /// factored flag (originally set to false, then reset to true after factorization), 
    /// notice we only want to factor once
    bool factored;  

  public:
    /// Constructor
    CsrMatrix(bool _trans, bool _sym,
	      IndexType_ _m, IndexType_ _n, IndexType_ _nnz,
        const cusparseMatDescr_t _descrA,
	      /*const*/ ValueType_ * _csrValA,
	      const IndexType_ * _csrRowPtrA,
	      const IndexType_ * _csrColIndA);

    /// Destructor
    virtual ~CsrMatrix();

    /// Get and Set CUDA stream    
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);  


    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
		    ValueType_ beta, ValueType_ * __restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;  

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const;         

    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  /// Graph Laplacian matrix
  template <typename IndexType_, typename ValueType_>
  class LaplacianMatrix 
    : public Matrix<IndexType_, ValueType_> {

  private:
    /// Adjacency matrix
    /*const*/ Matrix<IndexType_, ValueType_> * A;
    /// Degree of each vertex
    Vector<ValueType_> D;
    /// Preconditioning matrix
    Matrix<IndexType_, ValueType_> * M;  

  public:
    /// Constructor
    LaplacianMatrix(/*const*/ Matrix<IndexType_,ValueType_> & _A);

    /// Destructor
    virtual ~LaplacianMatrix();

    /// Get and Set CUDA stream    
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);   

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
		    ValueType_ beta, ValueType_ * __restrict__ y) const;
     /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;

    /// Scale a set of k vectors by a diagonal
    virtual void dm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;    

    /// Solve preconditioned system M x = f for a set of k vectors 
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const;    
    
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

    ///  Modularity matrix
  template <typename IndexType_, typename ValueType_>
  class ModularityMatrix 
    : public Matrix<IndexType_, ValueType_> {

  private:
    /// Adjacency matrix
    /*const*/ Matrix<IndexType_, ValueType_> * A;
    /// Degree of each vertex
    Vector<ValueType_> D;
    IndexType_ nnz;
    ValueType_ edge_sum;
    
    /// Preconditioning matrix
    Matrix<IndexType_, ValueType_> * M;  

  public:
    /// Constructor
    ModularityMatrix(/*const*/ Matrix<IndexType_,ValueType_> & _A, IndexType_ _nnz);

    /// Destructor
    virtual ~ModularityMatrix();

    /// Get and Set CUDA stream    
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);   

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
        ValueType_ beta, ValueType_ * __restrict__ y) const;
     /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;

    /// Scale a set of k vectors by a diagonal
    virtual void dm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;    

    /// Solve preconditioned system M x = f for a set of k vectors 
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const;    
   
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

// =============================================
// CUDA kernels
// =============================================

namespace {

/// Apply diagonal matrix to vector
template <typename IndexType_, typename ValueType_>
static __global__ void diagmv(IndexType_ n,
                              ValueType_ alpha,
                              const ValueType_ *__restrict__ D,
                              const ValueType_ *__restrict__ x,
                              ValueType_ *__restrict__ y)
{
  IndexType_ i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] += alpha * D[i] * x[i];
    i += blockDim.x * gridDim.x;
  }
}

/// Apply diagonal matrix to a set of dense vectors (tall matrix)
template <typename IndexType_, typename ValueType_, bool beta_is_zero>
static __global__ void diagmm(IndexType_ n,
                              IndexType_ k,
                              ValueType_ alpha,
                              const ValueType_ *__restrict__ D,
                              const ValueType_ *__restrict__ x,
                              ValueType_ beta,
                              ValueType_ *__restrict__ y)
{
  IndexType_ i, j, index;

  for (j = threadIdx.y + blockIdx.y * blockDim.y; j < k; j += blockDim.y * gridDim.y) {
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
      index = i + j * n;
      if (beta_is_zero) {
        y[index] = alpha * D[i] * x[index];
      } else {
        y[index] = alpha * D[i] * x[index] + beta * y[index];
      }
    }
  }
}
}  // namespace

// =============================================
// Dense matrix class
// =============================================

/// Constructor for dense matrix class
/** @param _trans Whether to transpose matrix.
 *  @param _m Number of rows.
 *  @param _n Number of columns.
 *  @param _A (Input, device memory, _m*_n entries) Matrix
 *    entries, stored column-major.
 *  @param _lda Leading dimension of _A.
 */
template <typename IndexType_, typename ValueType_>
DenseMatrix<IndexType_, ValueType_>::DenseMatrix(
  bool _trans, IndexType_ _m, IndexType_ _n, const ValueType_ *_A, IndexType_ _lda)
  : Matrix<IndexType_, ValueType_>(_m, _n), trans(_trans), A(_A), lda(_lda)
{
  Cublas::set_pointer_mode_host();
  if (_lda < _m) FatalError("invalid dense matrix parameter (lda<m)", NVGRAPH_ERR_BAD_PARAMETERS);
}

/// Destructor for dense matrix class
template <typename IndexType_, typename ValueType_>
DenseMatrix<IndexType_, ValueType_>::~DenseMatrix()
{
}

/// Get and Set CUDA stream
template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::setCUDAStream(cudaStream_t _s)
{
  this->s = _s;
  // printf("DenseMatrix setCUDAStream stream=%p\n",this->s);
  Cublas::setStream(_s);
}
template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::getCUDAStream(cudaStream_t *_s)
{
  *_s = this->s;
  // CHECK_CUBLAS(cublasGetStream(cublasHandle, _s));
}

/// Matrix-vector product for dense matrix class
/** y is overwritten with alpha*A*x+beta*y.
 *
 *  @param alpha Scalar.
 *  @param x (Input, device memory, n entries) Vector.
 *  @param beta Scalar.
 *  @param y (Input/output, device memory, m entries) Output vector.
 */
template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::mv(ValueType_ alpha,
                                             const ValueType_ *__restrict__ x,
                                             ValueType_ beta,
                                             ValueType_ *__restrict__ y) const
{
  Cublas::gemv(this->trans, this->m, this->n, &alpha, this->A, this->lda, x, 1, &beta, y, 1);
}

template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::mm(IndexType_ k,
                                             ValueType_ alpha,
                                             const ValueType_ *__restrict__ x,
                                             ValueType_ beta,
                                             ValueType_ *__restrict__ y) const
{
  Cublas::gemm(
    this->trans, false, this->m, k, this->n, &alpha, A, lda, x, this->m, &beta, y, this->n);
}

/// Color and Reorder
template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::color(IndexType_ *c, IndexType_ *p) const
{
}

template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::reorder(IndexType_ *p) const
{
}

/// Incomplete Cholesky (setup, factor and solve)
template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::prec_setup(Matrix<IndexType_, ValueType_> *_M)
{
  printf("ERROR: DenseMatrix prec_setup dispacthed\n");
  // exit(1);
}

template <typename IndexType_, typename ValueType_>
void DenseMatrix<IndexType_, ValueType_>::prec_solve(IndexType_ k,
                                                     ValueType_ alpha,
                                                     ValueType_ *__restrict__ fx,
                                                     ValueType_ *__restrict__ t) const
{
  printf("ERROR: DenseMatrix prec_solve dispacthed\n");
  // exit(1);
}

template <typename IndexType_, typename ValueType_>
ValueType_ DenseMatrix<IndexType_, ValueType_>::getEdgeSum() const
{
  return 0.0;
}

// =============================================
// CSR matrix class
// =============================================

/// Constructor for CSR matrix class
/** @param _transA Whether to transpose matrix.
 *  @param _m Number of rows.
 *  @param _n Number of columns.
 *  @param _nnz Number of non-zero entries.
 *  @param _descrA Matrix properties.
 *  @param _csrValA (Input, device memory, _nnz entries) Matrix
 *    entry values.
 *  @param _csrRowPtrA (Input, device memory, _m+1 entries) Pointer
 *    to first entry in each row.
 *  @param _csrColIndA (Input, device memory, _nnz entries) Column
 *    index of each matrix entry.
 */
template <typename IndexType_, typename ValueType_>
CsrMatrix<IndexType_, ValueType_>::CsrMatrix(bool _trans,
                                             bool _sym,
                                             IndexType_ _m,
                                             IndexType_ _n,
                                             IndexType_ _nnz,
                                             const cusparseMatDescr_t _descrA,
                                             /*const*/ ValueType_ *_csrValA,
                                             const IndexType_ *_csrRowPtrA,
                                             const IndexType_ *_csrColIndA)
  : Matrix<IndexType_, ValueType_>(_m, _n),
    trans(_trans),
    sym(_sym),
    nnz(_nnz),
    descrA(_descrA),
    csrValA(_csrValA),
    csrRowPtrA(_csrRowPtrA),
    csrColIndA(_csrColIndA)
{
  if (nnz < 0) FatalError("invalid CSR matrix parameter (nnz<0)", NVGRAPH_ERR_BAD_PARAMETERS);
  Cusparse::set_pointer_mode_host();
}

/// Destructor for CSR matrix class
template <typename IndexType_, typename ValueType_>
CsrMatrix<IndexType_, ValueType_>::~CsrMatrix()
{
}

/// Get and Set CUDA stream
template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::setCUDAStream(cudaStream_t _s)
{
  this->s = _s;
  // printf("CsrMatrix setCUDAStream stream=%p\n",this->s);
  Cusparse::setStream(_s);
}
template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::getCUDAStream(cudaStream_t *_s)
{
  *_s = this->s;
  // CHECK_CUSPARSE(cusparseGetStream(Cusparse::get_handle(), _s));
}
template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::mm(IndexType_ k,
                                           ValueType_ alpha,
                                           const ValueType_ *__restrict__ x,
                                           ValueType_ beta,
                                           ValueType_ *__restrict__ y) const
{
  // CHECK_CUSPARSE(cusparseXcsrmm(Cusparse::get_handle(), transA, this->m, k, this->n, nnz, &alpha,
  // descrA, csrValA, csrRowPtrA, csrColIndA, x, this->n, &beta, y, this->m));
  Cusparse::csrmm(this->trans,
                  this->sym,
                  this->m,
                  k,
                  this->n,
                  this->nnz,
                  &alpha,
                  this->csrValA,
                  this->csrRowPtrA,
                  this->csrColIndA,
                  x,
                  this->n,
                  &beta,
                  y,
                  this->m);
}

/// Color and Reorder
template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::color(IndexType_ *c, IndexType_ *p) const
{
}

template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::reorder(IndexType_ *p) const
{
}

/// Incomplete Cholesky (setup, factor and solve)
template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::prec_setup(Matrix<IndexType_, ValueType_> *_M)
{
  // printf("CsrMatrix prec_setup dispacthed\n");
  if (!factored) {
    // analyse lower triangular factor
    CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_l));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT));
    CHECK_CUSPARSE(cusparseXcsrsm_analysis(Cusparse::get_handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           this->m,
                                           nnz,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           info_l));
    // analyse upper triangular factor
    CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_u));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseXcsrsm_analysis(Cusparse::get_handle(),
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           this->m,
                                           nnz,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           info_u));
    // perform csrilu0 (should be slightly faster than csric0)
    CHECK_CUSPARSE(cusparseXcsrilu0(Cusparse::get_handle(),
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    this->m,
                                    descrA,
                                    csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    info_l));
    // set factored flag to true
    factored = true;
  }
}

template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::prec_solve(IndexType_ k,
                                                   ValueType_ alpha,
                                                   ValueType_ *__restrict__ fx,
                                                   ValueType_ *__restrict__ t) const
{
  // printf("CsrMatrix prec_solve dispacthed (stream %p)\n",this->s);

  // preconditioning Mx=f (where M = L*U, threfore x=U\(L\f))
  // solve lower triangular factor
  CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER));
  CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_UNIT));
  CHECK_CUSPARSE(cusparseXcsrsm_solve(Cusparse::get_handle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      this->m,
                                      k,
                                      alpha,
                                      descrA,
                                      csrValA,
                                      csrRowPtrA,
                                      csrColIndA,
                                      info_l,
                                      fx,
                                      this->m,
                                      t,
                                      this->m));
  // solve upper triangular factor
  CHECK_CUSPARSE(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER));
  CHECK_CUSPARSE(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));
  CHECK_CUSPARSE(cusparseXcsrsm_solve(Cusparse::get_handle(),
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      this->m,
                                      k,
                                      alpha,
                                      descrA,
                                      csrValA,
                                      csrRowPtrA,
                                      csrColIndA,
                                      info_u,
                                      t,
                                      this->m,
                                      fx,
                                      this->m));
}

/// Matrix-vector product for CSR matrix class
/** y is overwritten with alpha*A*x+beta*y.
 *
 *  @param alpha Scalar.
 *  @param x (Input, device memory, n entries) Vector.
 *  @param beta Scalar.
 *  @param y (Input/output, device memory, m entries) Output vector.
 */
template <typename IndexType_, typename ValueType_>
void CsrMatrix<IndexType_, ValueType_>::mv(ValueType_ alpha,
                                           const ValueType_ *__restrict__ x,
                                           ValueType_ beta,
                                           ValueType_ *__restrict__ y) const
{
  // TODO: consider using merge-path csrmv
  Cusparse::csrmv(this->trans,
                  this->sym,
                  this->m,
                  this->n,
                  this->nnz,
                  &alpha,
                  this->csrValA,
                  this->csrRowPtrA,
                  this->csrColIndA,
                  x,
                  &beta,
                  y);
}

template <typename IndexType_, typename ValueType_>
ValueType_ CsrMatrix<IndexType_, ValueType_>::getEdgeSum() const
{
  return 0.0;
}

// =============================================
// Laplacian matrix class
// =============================================

/// Constructor for Laplacian matrix class
/** @param A Adjacency matrix
 */
template <typename IndexType_, typename ValueType_>
LaplacianMatrix<IndexType_, ValueType_>::LaplacianMatrix(
  /*const*/ Matrix<IndexType_, ValueType_> &_A)
  : Matrix<IndexType_, ValueType_>(_A.m, _A.n), A(&_A)
{
  // Check that adjacency matrix is square
  if (_A.m != _A.n)
    FatalError("cannot construct Laplacian matrix from non-square adjacency matrix",
               NVGRAPH_ERR_BAD_PARAMETERS);
  // set CUDA stream
  this->s = NULL;
  // Construct degree matrix
  D.allocate(_A.m, this->s);
  Vector<ValueType_> ones(this->n, this->s);
  ones.fill(1.0);
  _A.mv(1, ones.raw(), 0, D.raw());

  // Set preconditioning matrix pointer to NULL
  M = NULL;
}

/// Destructor for Laplacian matrix class
template <typename IndexType_, typename ValueType_>
LaplacianMatrix<IndexType_, ValueType_>::~LaplacianMatrix()
{
}

/// Get and Set CUDA stream
template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::setCUDAStream(cudaStream_t _s)
{
  this->s = _s;
  // printf("LaplacianMatrix setCUDAStream stream=%p\n",this->s);
  A->setCUDAStream(_s);
  if (M != NULL) { M->setCUDAStream(_s); }
}
template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::getCUDAStream(cudaStream_t *_s)
{
  *_s = this->s;
  // A->getCUDAStream(_s);
}

/// Matrix-vector product for Laplacian matrix class
/** y is overwritten with alpha*A*x+beta*y.
 *
 *  @param alpha Scalar.
 *  @param x (Input, device memory, n entries) Vector.
 *  @param beta Scalar.
 *  @param y (Input/output, device memory, m entries) Output vector.
 */
template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::mv(ValueType_ alpha,
                                                 const ValueType_ *__restrict__ x,
                                                 ValueType_ beta,
                                                 ValueType_ *__restrict__ y) const
{
  // Scale result vector
  if (beta == 0)
    CHECK_CUDA(cudaMemset(y, 0, (this->n) * sizeof(ValueType_)))
  else if (beta != 1)
    thrust::transform(thrust::device_pointer_cast(y),
                      thrust::device_pointer_cast(y + this->n),
                      thrust::make_constant_iterator(beta),
                      thrust::device_pointer_cast(y),
                      thrust::multiplies<ValueType_>());

  // Apply diagonal matrix
  dim3 gridDim, blockDim;
  gridDim.x  = min(((this->n) + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
  gridDim.y  = 1;
  gridDim.z  = 1;
  blockDim.x = BLOCK_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  diagmv<<<gridDim, blockDim, 0, A->s>>>(this->n, alpha, D.raw(), x, y);
  cudaCheckError();

  // Apply adjacency matrix
  A->mv(-alpha, x, 1, y);
}
/// Matrix-vector product for Laplacian matrix class
/** y is overwritten with alpha*A*x+beta*y.
 *
 *  @param alpha Scalar.
 *  @param x (Input, device memory, n*k entries) nxk dense matrix.
 *  @param beta Scalar.
 *  @param y (Input/output, device memory, m*k entries) Output mxk dense matrix.
 */
template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::mm(IndexType_ k,
                                                 ValueType_ alpha,
                                                 const ValueType_ *__restrict__ x,
                                                 ValueType_ beta,
                                                 ValueType_ *__restrict__ y) const
{
  // Apply diagonal matrix
  ValueType_ one = (ValueType_)1.0;
  this->dm(k, alpha, x, beta, y);

  // Apply adjacency matrix
  A->mm(k, -alpha, x, one, y);
}

template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::dm(IndexType_ k,
                                                 ValueType_ alpha,
                                                 const ValueType_ *__restrict__ x,
                                                 ValueType_ beta,
                                                 ValueType_ *__restrict__ y) const
{
  IndexType_ t = k * (this->n);
  dim3 gridDim, blockDim;

  // setup launch parameters
  gridDim.x  = min(((this->n) + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
  gridDim.y  = min(k, 65535);
  gridDim.z  = 1;
  blockDim.x = BLOCK_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;

  // Apply diagonal matrix
  if (beta == 0.0) {
    // set vectors to 0 (WARNING: notice that you need to set, not scale, because of NaNs corner
    // case)
    CHECK_CUDA(cudaMemset(y, 0, t * sizeof(ValueType_)));
    diagmm<IndexType_, ValueType_, true>
      <<<gridDim, blockDim, 0, A->s>>>(this->n, k, alpha, D.raw(), x, beta, y);
  } else {
    diagmm<IndexType_, ValueType_, false>
      <<<gridDim, blockDim, 0, A->s>>>(this->n, k, alpha, D.raw(), x, beta, y);
  }
  cudaCheckError();
}

/// Color and Reorder
template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::color(IndexType_ *c, IndexType_ *p) const
{
}

template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::reorder(IndexType_ *p) const
{
}

/// Solve preconditioned system M x = f for a set of k vectors
template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::prec_setup(Matrix<IndexType_, ValueType_> *_M)
{
  // save the pointer to preconditioner M
  M = _M;
  if (M != NULL) {
    // setup the preconditioning matrix M
    M->prec_setup(NULL);
  }
}

template <typename IndexType_, typename ValueType_>
void LaplacianMatrix<IndexType_, ValueType_>::prec_solve(IndexType_ k,
                                                         ValueType_ alpha,
                                                         ValueType_ *__restrict__ fx,
                                                         ValueType_ *__restrict__ t) const
{
  if (M != NULL) {
    // preconditioning
    M->prec_solve(k, alpha, fx, t);
  }
}

template <typename IndexType_, typename ValueType_>
ValueType_ LaplacianMatrix<IndexType_, ValueType_>::getEdgeSum() const
{
  return 0.0;
}
// =============================================
// Modularity matrix class
// =============================================

/// Constructor for Modularity matrix class
/** @param A Adjacency matrix
 */
template <typename IndexType_, typename ValueType_>
ModularityMatrix<IndexType_, ValueType_>::ModularityMatrix(
  /*const*/ Matrix<IndexType_, ValueType_> &_A, IndexType_ _nnz)
  : Matrix<IndexType_, ValueType_>(_A.m, _A.n), A(&_A), nnz(_nnz)
{
  // Check that adjacency matrix is square
  if (_A.m != _A.n)
    FatalError("cannot construct Modularity matrix from non-square adjacency matrix",
               NVGRAPH_ERR_BAD_PARAMETERS);

  // set CUDA stream
  this->s = NULL;
  // Construct degree matrix
  D.allocate(_A.m, this->s);
  Vector<ValueType_> ones(this->n, this->s);
  ones.fill(1.0);
  _A.mv(1, ones.raw(), 0, D.raw());
  // D.dump(0,this->n);
  edge_sum = D.nrm1();

  // Set preconditioning matrix pointer to NULL
  M = NULL;
}

/// Destructor for Modularity matrix class
template <typename IndexType_, typename ValueType_>
ModularityMatrix<IndexType_, ValueType_>::~ModularityMatrix()
{
}

/// Get and Set CUDA stream
template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::setCUDAStream(cudaStream_t _s)
{
  this->s = _s;
  // printf("ModularityMatrix setCUDAStream stream=%p\n",this->s);
  A->setCUDAStream(_s);
  if (M != NULL) { M->setCUDAStream(_s); }
}

template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::getCUDAStream(cudaStream_t *_s)
{
  *_s = this->s;
  // A->getCUDAStream(_s);
}

/// Matrix-vector product for Modularity matrix class
/** y is overwritten with alpha*A*x+beta*y.
 *
 *  @param alpha Scalar.
 *  @param x (Input, device memory, n entries) Vector.
 *  @param beta Scalar.
 *  @param y (Input/output, device memory, m entries) Output vector.
 */
template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::mv(ValueType_ alpha,
                                                  const ValueType_ *__restrict__ x,
                                                  ValueType_ beta,
                                                  ValueType_ *__restrict__ y) const
{
  // Scale result vector
  if (alpha != 1 || beta != 0)
    FatalError("This isn't implemented for Modularity Matrix currently",
               NVGRAPH_ERR_NOT_IMPLEMENTED);

  // CHECK_CUBLAS(cublasXdot(handle, this->n, const double *x, int incx, const double *y, int incy,
  // double *result));
  // y = A*x
  A->mv(alpha, x, 0, y);
  ValueType_ dot_res;
  // gamma = d'*x
  Cublas::dot(this->n, D.raw(), 1, x, 1, &dot_res);
  // y = y -(gamma/edge_sum)*d
  Cublas::axpy(this->n, -(dot_res / this->edge_sum), D.raw(), 1, y, 1);
}
/// Matrix-vector product for Modularity matrix class
/** y is overwritten with alpha*A*x+beta*y.
 *
 *  @param alpha Scalar.
 *  @param x (Input, device memory, n*k entries) nxk dense matrix.
 *  @param beta Scalar.
 *  @param y (Input/output, device memory, m*k entries) Output mxk dense matrix.
 */
template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::mm(IndexType_ k,
                                                  ValueType_ alpha,
                                                  const ValueType_ *__restrict__ x,
                                                  ValueType_ beta,
                                                  ValueType_ *__restrict__ y) const
{
  FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
}

template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::dm(IndexType_ k,
                                                  ValueType_ alpha,
                                                  const ValueType_ *__restrict__ x,
                                                  ValueType_ beta,
                                                  ValueType_ *__restrict__ y) const
{
  FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
}

/// Color and Reorder
template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::color(IndexType_ *c, IndexType_ *p) const
{
  FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
}

template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::reorder(IndexType_ *p) const
{
  FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
}

/// Solve preconditioned system M x = f for a set of k vectors
template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::prec_setup(Matrix<IndexType_, ValueType_> *_M)
{
  // save the pointer to preconditioner M
  M = _M;
  if (M != NULL) {
    // setup the preconditioning matrix M
    M->prec_setup(NULL);
  }
}

template <typename IndexType_, typename ValueType_>
void ModularityMatrix<IndexType_, ValueType_>::prec_solve(IndexType_ k,
                                                          ValueType_ alpha,
                                                          ValueType_ *__restrict__ fx,
                                                          ValueType_ *__restrict__ t) const
{
  if (M != NULL) {
    FatalError("This isn't implemented for Modularity Matrix currently",
               NVGRAPH_ERR_NOT_IMPLEMENTED);
  }
}

template <typename IndexType_, typename ValueType_>
ValueType_ ModularityMatrix<IndexType_, ValueType_>::getEdgeSum() const
{
  return edge_sum;
}

} // namespace matrix
} // namespace raft
