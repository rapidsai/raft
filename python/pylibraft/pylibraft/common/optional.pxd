# polyfill for libcpp.optional
#
# We're still using cython v0.29.x - which doesn't have std::optional
# support. Include this definition here as suggested by
# https://github.com/cython/cython/issues/3293#issuecomment-1223058101

from libcpp cimport bool

cdef extern from "<optional>" namespace "std" nogil:
    cdef cppclass nullopt_t:
        nullopt_t()

    cdef nullopt_t nullopt

    cdef cppclass optional[T]:
        ctypedef T value_type
        optional()
        optional(nullopt_t)
        optional(optional&) except +
        optional(T&) except +
        bool has_value()
        T& value()
        T& value_or[U](U& default_value)
        void swap(optional&)
        void reset()
        T& emplace(...)
        T& operator*()
        #T* operator->() # Not Supported
        optional& operator=(optional&)
        optional& operator=[U](U&)
        bool operator bool()
        bool operator!()
        bool operator==[U](optional&, U&)
        bool operator!=[U](optional&, U&)
        bool operator<[U](optional&, U&)
        bool operator>[U](optional&, U&)
        bool operator<=[U](optional&, U&)
        bool operator>=[U](optional&, U&)

    optional[T] make_optional[T](...) except +
