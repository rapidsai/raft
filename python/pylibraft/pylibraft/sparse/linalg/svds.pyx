#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import warnings

import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint32_t, uint64_t, uintptr_t

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources


cdef extern from "raft/sparse/solver/svds_config.hpp" \
        namespace "raft::sparse::solver" nogil:

    cdef cppclass sparse_svd_config[ValueTypeT]:
        int n_components
        int n_oversamples
        int n_power_iters
        optional[uint64_t] seed


cdef extern from "raft_runtime/solver/randomized_svds.hpp" \
        namespace "raft::runtime::solver" nogil:

    cdef void sparse_randomized_svd_float(
        const device_resources &handle,
        const sparse_svd_config[float] &config,
        device_vector_view[int, uint32_t] indptr,
        device_vector_view[int, uint32_t] indices,
        device_vector_view[float, uint32_t] data,
        int n_rows, int n_cols, int nnz,
        device_vector_view[float, uint32_t] singular_values,
        device_matrix_view[float, uint32_t, col_major] U,
        device_matrix_view[float, uint32_t, col_major] Vt) except +

    cdef void sparse_randomized_svd_double(
        const device_resources &handle,
        const sparse_svd_config[double] &config,
        device_vector_view[int, uint32_t] indptr,
        device_vector_view[int, uint32_t] indices,
        device_vector_view[double, uint32_t] data,
        int n_rows, int n_cols, int nnz,
        device_vector_view[double, uint32_t] singular_values,
        device_matrix_view[double, uint32_t, col_major] U,
        device_matrix_view[double, uint32_t, col_major] Vt) except +


@auto_sync_handle
def svds(A, k=6, n_oversamples=10, n_power_iters=2,
         seed=None, handle=None):
    """
    Compute the largest ``k`` singular values and corresponding singular
    vectors of a sparse matrix using randomized SVD.

    Computes the truncated SVD: ``A ~ U @ diag(S) @ Vt``.

    Args:
        A (cupyx.scipy.sparse.csr_matrix): Sparse CSR matrix of shape
            ``(m, n)``. Must be of type
            :class:`cupyx.scipy.sparse.csr_matrix`.
        k (int): Number of singular values and vectors to compute. Must be
            ``1 <= k < min(m, n)``. Default 6.
        n_oversamples (int): Number of extra random vectors for better
            approximation. Total subspace dimension is ``k + n_oversamples``.
            Default 10.
        n_power_iters (int): Number of power iteration passes. More
            iterations improve accuracy for matrices with slowly decaying
            singular values. Default 2.
        seed (int or None): Random seed for reproducibility. If ``None``,
            a non-deterministic seed is used.
        handle: RAFT resource handle. If ``None``, a default is created.

    Returns:
        tuple:
            ``(U, S, Vt)`` where ``U`` is left singular vectors ``(m, k)``,
            ``S`` is singular values ``(k,)`` in descending order, and
            ``Vt`` is right singular vectors ``(k, n)``.

    .. seealso::
        :func:`scipy.sparse.linalg.svds`

    .. note::
        This function uses randomized SVD (Halko et al. 2009) with
        CholeskyQR2 orthogonalization (Tomas et al. 2024) for efficient
        GPU execution.

    """

    if A is None:
        raise ValueError("'A' cannot be None!")

    if not cupy_sparse.issparse(A):
        raise TypeError(
            "Expected a cupyx.scipy.sparse matrix, got %s" % type(A).__name__
        )

    if not isinstance(A, cupy_sparse.csr_matrix):
        raise TypeError(
            "Expected a cupyx.scipy.sparse.csr_matrix, got %s. "
            "Use A.tocsr() to convert." % type(A).__name__
        )

    m, n = A.shape

    if k < 1 or k >= min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k < min(m, n), got k={k}, m={m}, n={n}"
        )

    # Extract CSR arrays and ensure int32 indices
    indptr = A.indptr
    indices = A.indices
    data = A.data

    IndexType = indptr.dtype
    if IndexType == np.int64:
        warnings.warn(
            "Input matrix has int64 indices which will be converted to "
            "int32. This may cause issues if index values exceed 2^31 - 1.",
            UserWarning,
            stacklevel=2,
        )
        indptr = indptr.astype(np.int32)
        indices = indices.astype(np.int32)
    elif IndexType != np.int32:
        raise TypeError("dtype IndexType=%s not supported, "
                        "expected int32 or int64" % IndexType)

    indptr = cai_wrapper(indptr)
    indices = cai_wrapper(indices)
    data = cai_wrapper(data)

    ValueType = data.dtype
    nnz = A.nnz

    indptr_ptr = <uintptr_t>indptr.data
    indices_ptr = <uintptr_t>indices.data
    data_ptr = <uintptr_t>data.data

    cdef optional[uint64_t] seed_opt
    if seed is not None:
        seed_opt = <uint64_t>seed

    # Allocate outputs
    S_out = device_ndarray.empty((k,), dtype=ValueType, order='F')
    U_out = device_ndarray.empty((m, k), dtype=ValueType, order='F')
    Vt_out = device_ndarray.empty((k, n), dtype=ValueType, order='F')

    S_cai = cai_wrapper(S_out)
    U_cai = cai_wrapper(U_out)
    Vt_cai = cai_wrapper(Vt_out)

    S_ptr = <uintptr_t>S_cai.data
    U_ptr = <uintptr_t>U_cai.data
    Vt_ptr = <uintptr_t>Vt_cai.data

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    cdef sparse_svd_config[float] cfg_float
    cdef sparse_svd_config[double] cfg_double

    if ValueType == np.float32:
        cfg_float.n_components = k
        cfg_float.n_oversamples = n_oversamples
        cfg_float.n_power_iters = n_power_iters
        cfg_float.seed = seed_opt
        sparse_randomized_svd_float(
            deref(h),
            cfg_float,
            make_device_vector_view(<int *>indptr_ptr, <uint32_t>(m + 1)),
            make_device_vector_view(<int *>indices_ptr, <uint32_t>nnz),
            make_device_vector_view(<float *>data_ptr, <uint32_t>nnz),
            m, n, nnz,
            make_device_vector_view(<float *>S_ptr, <uint32_t>k),
            make_device_matrix_view[float, uint32_t, col_major](
                <float *>U_ptr, <uint32_t>m, <uint32_t>k),
            make_device_matrix_view[float, uint32_t, col_major](
                <float *>Vt_ptr, <uint32_t>k, <uint32_t>n),
        )
    elif ValueType == np.float64:
        cfg_double.n_components = k
        cfg_double.n_oversamples = n_oversamples
        cfg_double.n_power_iters = n_power_iters
        cfg_double.seed = seed_opt
        sparse_randomized_svd_double(
            deref(h),
            cfg_double,
            make_device_vector_view(<int *>indptr_ptr, <uint32_t>(m + 1)),
            make_device_vector_view(<int *>indices_ptr, <uint32_t>nnz),
            make_device_vector_view(<double *>data_ptr, <uint32_t>nnz),
            m, n, nnz,
            make_device_vector_view(<double *>S_ptr, <uint32_t>k),
            make_device_matrix_view[double, uint32_t, col_major](
                <double *>U_ptr, <uint32_t>m, <uint32_t>k),
            make_device_matrix_view[double, uint32_t, col_major](
                <double *>Vt_ptr, <uint32_t>k, <uint32_t>n),
        )
    else:
        raise TypeError("dtype ValueType=%s not supported, "
                        "expected float32 or float64" % ValueType)

    return (cp.asarray(U_out), cp.asarray(S_out), cp.asarray(Vt_out))
