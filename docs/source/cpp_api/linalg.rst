Linear Algebra
==============

This page provides C++ class references for the publicly-exposed elements of the `raft/linalg` (dense) linear algebra headers.
In addition to providing highly optimized arithmetic and matrix/vector operations, RAFT provides a consistent user experience
by providing common BLAS routines, standard linear system solvers, factorization and eigenvalue solvers. Some of these routines
hide the complexities of lower-level C-based libraries provided in the CUDA toolkit 

.. role:: py(code)
   :language: c++
   :class: highlight

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   linalg_arithmetic.rst
   linalg_blas.rst
   linalg_map_reduce.rst
   linalg_matrix.rst
   linalg_matrix_vector.rst
   linalg_solver.rst