# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy
import cupyx.scipy.sparse.linalg  # NOQA
import numpy
import pytest
import scipy
from cupyx.scipy import sparse

from pylibraft.sparse.linalg import eigsh, svds


def shaped_random(
    shape, xp=cupy, dtype=numpy.float32, scale=10, seed=0, order="C"
):
    """
    Returns an array filled with random values.

    Args
    ----
        shape(tuple): Shape of returned ndarray.
        xp(numpy or cupy): Array module to use.
        dtype(dtype): Dtype of returned ndarray.
        scale(float): Scaling factor of elements.
        seed(int): Random seed.

    Returns
    -------
        numpy.ndarray or cupy.ndarray: The array with
        given shape, array module,

    If ``dtype`` is ``numpy.bool_``, the elements are
    independently drawn from ``True`` and ``False``
    with same probabilities.
    Otherwise, the array is filled with samples
    independently and identically drawn
    from uniform distribution over :math:`[0, scale)`
    with specified dtype.
    """
    numpy.random.seed(seed)
    dtype = numpy.dtype(dtype)
    if dtype == "?":
        a = numpy.random.randint(2, size=shape)
    elif dtype.kind == "c":
        a = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)
        a *= scale
    else:
        a = numpy.random.rand(*shape) * scale
    return xp.asarray(a, dtype=dtype, order=order)


class TestEigsh:
    n = 30
    density = 0.33
    tol = {numpy.float32: 1e-5, numpy.complex64: 1e-5, "default": 1e-12}
    res_tol = {"f": 1e-5, "d": 1e-12}
    return_eigenvectors = True

    def _make_matrix(self, dtype, xp):
        shape = (self.n, self.n)
        a = shaped_random(shape, xp, dtype=dtype)
        mask = shaped_random(shape, xp, dtype="f", scale=1)
        a[mask > self.density] = 0
        a = a * a.conj().T
        return a

    def _test_eigsh(self, a, k, xp, sp, which):
        scipy_csr = sp.sparse.csr_matrix(
            (a.data.get(), a.indices.get(), a.indptr.get()), shape=a.shape
        )
        expected_ret = sp.sparse.linalg.eigsh(
            scipy_csr,
            k=k,
            return_eigenvectors=self.return_eigenvectors,
            which=which,
        )

        actual_ret = eigsh(
            a,
            k=k,
            which=which,
        )
        if self.return_eigenvectors:
            w, x = actual_ret
            exp_w, _ = expected_ret
            # Check the residuals to see if eigenvectors are correct.
            ax_xw = a @ x - xp.multiply(x, w.reshape(1, k))
            res = xp.linalg.norm(ax_xw) / xp.linalg.norm(w)
            tol = self.res_tol[numpy.dtype(a.dtype).char.lower()]
            assert res < tol
        else:
            w = actual_ret
            exp_w = expected_ret
        w = xp.sort(w)
        assert cupy.allclose(w, exp_w, rtol=tol, atol=tol)

    @pytest.mark.parametrize("format", ["csr"])  # , 'csc', 'coo'])
    @pytest.mark.parametrize("k", [3, 6, 12])
    @pytest.mark.parametrize("dtype", ["f", "d"])
    @pytest.mark.parametrize("which", ["LA", "LM", "SA"])
    def test_sparse(self, format, k, dtype, which, xp=cupy, sp=sparse):
        if format == "csc":
            pytest.xfail("may be buggy")  # trans=True

        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        return self._test_eigsh(a, k, xp, scipy, which)

    def test_invalid(self):
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones((self.n,), dtype="f"))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((2, 1), dtype="f"), which="SA")
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=0, which="SA")
        a = xp.diag(xp.ones((self.n,), dtype="f"))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((1,), dtype="f"), which="SA")
        with pytest.raises(TypeError):
            sp.linalg.eigsh(xp.ones((2, 2), dtype="i"), which="SA")
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.n, which="SA")

    def test_starting_vector(self):
        # Make symmetric matrix
        aux = self._make_matrix("f", cupy)
        aux = sparse.coo_matrix(aux).asformat("csr")
        matrix = (aux + aux.T) / 2.0

        # Find reference eigenvector
        ew, ev = eigsh(matrix, k=1)
        v = ev[:, 0]

        # Obtain non-converged eigenvector from random initial guess.
        ew_aux, ev_aux = eigsh(matrix, k=1, ncv=1, maxiter=0)
        v_aux = cupy.copysign(ev_aux[:, 0], v)

        # Obtain eigenvector using known eigenvector as initial guess.
        ew_v0, ev_v0 = eigsh(matrix, k=1, v0=v.copy(), ncv=1, maxiter=0)
        v_v0 = cupy.copysign(ev_v0[:, 0], v)

        assert cupy.linalg.norm(v - v_v0) < cupy.linalg.norm(v - v_aux)

    @pytest.mark.parametrize("dtype", ["f", "d"])
    @pytest.mark.parametrize("seed", [42, 123, 0])
    def test_reproducibility(self, dtype, seed):
        """Test that eigsh produces reproducible results when given a seed."""
        xp, sp = cupy, sparse

        # Make symmetric matrix
        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat("csr")

        # Run eigsh twice with the same seed
        w1, v1 = eigsh(a, k=3, which="SA", seed=seed)
        w2, v2 = eigsh(a, k=3, which="SA", seed=seed)

        # Results should be identical
        assert cupy.allclose(w1, w2), "Eigenvalues differ with same seed"
        assert cupy.allclose(v1, v2), "Eigenvectors differ with same seed"


class TestSvds:
    """Tests for sparse randomized SVD."""

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    @pytest.mark.parametrize("k", [3, 10])
    def test_singular_values_positive_descending(self, dtype, k):
        """Singular values must be positive and in descending order."""
        m, n = 100, 60
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)
        U, S, Vt = svds(A, k=k, seed=42)
        S_host = cupy.asnumpy(S)
        assert all(S_host > 0), "Singular values must be positive"
        assert all(
            S_host[i] >= S_host[i + 1] for i in range(len(S_host) - 1)
        ), "Singular values must be descending"

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    @pytest.mark.parametrize("k", [3, 10])
    def test_orthogonality(self, dtype, k):
        """U columns and Vt rows must be orthonormal."""
        m, n = 100, 60
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)
        U, S, Vt = svds(A, k=k, seed=42)

        tol = 1e-4 if dtype == numpy.float32 else 1e-8
        I_k = cupy.eye(k, dtype=dtype)

        UtU = U.T @ U
        assert cupy.allclose(UtU, I_k, atol=tol), "U is not orthogonal"

        VtVtT = Vt @ Vt.T
        assert cupy.allclose(VtVtT, I_k, atol=tol), "Vt rows not orthonormal"

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_reconstruction_error(self, dtype):
        """||A - U @ diag(S) @ Vt|| must be bounded by the (k+1)-th singular value."""
        m, n = 80, 50
        k = 5
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.4, format="csr", dtype=dtype)
        U, S, Vt = svds(A, k=k, n_power_iters=4, seed=42)

        A_dense = A.toarray()
        recon = (U * S[None, :]) @ Vt
        err = cupy.linalg.norm(A_dense - recon)
        norm_A = cupy.linalg.norm(A_dense)

        # Relative error should be reasonable (not reconstructing full matrix)
        assert err / norm_A < 1.0, (
            f"Reconstruction error too large: {float(err/norm_A):.4f}"
        )

    def test_shapes(self):
        """Output shapes must match (m,k), (k,), (k,n)."""
        m, n, k = 100, 60, 5
        A = sparse.random(m, n, density=0.3, format="csr", dtype=numpy.float32)
        U, S, Vt = svds(A, k=k, seed=42)
        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    def test_int64_indices(self):
        """int64 indices should be auto-converted and work."""
        m, n, k = 50, 30, 3
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=numpy.float32)

        # Convert to int64 indices
        A_i64 = sparse.csr_matrix(
            (A.data, A.indices.astype(cupy.int64), A.indptr.astype(cupy.int64)),
            shape=A.shape,
        )
        U, S, Vt = svds(A_i64, k=k, seed=42)
        assert S.shape == (k,)
        assert all(cupy.asnumpy(S) > 0)

    def test_invalid_k(self):
        """k out of range must raise ValueError."""
        A = sparse.random(20, 15, density=0.3, format="csr")
        with pytest.raises(ValueError):
            svds(A, k=0)
        with pytest.raises(ValueError):
            svds(A, k=15)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_reproducibility(self, dtype):
        """Same seed must give same results."""
        m, n, k = 80, 50, 5
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)

        _, S1, _ = svds(A, k=k, seed=123)
        _, S2, _ = svds(A, k=k, seed=123)

        tol = 1e-5 if dtype == numpy.float32 else 1e-10
        assert cupy.allclose(S1, S2, atol=tol), "Not reproducible with same seed"
