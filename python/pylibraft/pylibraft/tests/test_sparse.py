# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy
import cupyx.scipy.sparse.linalg  # NOQA
import numpy
import pytest
import scipy
from cupyx.scipy import sparse

from pylibraft.sparse.linalg import eigsh


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
    res_tol_factor = {"SA": 1, "LA": 1, "LM": 1, "SM": 10}
    maxiter = 10000000
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
            maxiter=self.maxiter,
        )
        actual_ret = eigsh(
            a,
            k=k,
            which=which,
            maxiter=self.maxiter,
            v0=xp.ones((self.n,), dtype=a.dtype),
        )
        # cupy_actual_ret = sparse.linalg.eigsh(
        #     a, k=k, which=which, maxiter=self.maxiter
        # )
        if self.return_eigenvectors:
            w, x = actual_ret
            exp_w, _ = expected_ret
            # cupy_exp_w, _ = cupy_actual_ret
            # Check the residuals to see if eigenvectors are correct.
            ax_xw = a @ x - xp.multiply(x, w.reshape(1, k))
            res = xp.linalg.norm(ax_xw) / xp.linalg.norm(w)
            tol = (
                self.res_tol[numpy.dtype(a.dtype).char.lower()]
                * self.res_tol_factor[which]
            )
            assert res < tol
        else:
            w = actual_ret
            exp_w = expected_ret
        w = xp.sort(w)
        print(w, "raft")
        print(exp_w, "scipy")
        # print(cupy_exp_w, "cupy")
        assert cupy.allclose(w, exp_w, rtol=tol, atol=tol)

    @pytest.mark.parametrize("format", ["csr"])  # , 'csc', 'coo'])
    @pytest.mark.parametrize("k", [3, 6, 12])
    @pytest.mark.parametrize("dtype", ["f", "d"])
    @pytest.mark.parametrize("which", ["LA", "LM", "SA", "SM"])
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
