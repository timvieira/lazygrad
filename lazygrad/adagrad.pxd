#!python
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration


#cdef extern from "fastonebigheader.h" nogil:
#  float fastpow(float, float)

from libc.math cimport log, exp, sqrt, pow

#cdef inline float fastpow(float x, float y) nogil:
#    #return x ** y
#    return exp(y * log(x))


#cdef extern from "/home/timv/projects/lazygrad/lazygrad/invsqrt.h" nogil:
#    float invsqrt(float)
#    double fastpow(double, double)

cdef inline double invsqrt(double x) nogil:
    return 1/sqrt(x)
#    return x ** -0.5

cdef inline double fastpow(double x, double y) nogil:
    return pow(x, y)
#    return x ** y

cdef class LazyRegularizedAdagrad:

    cdef public:
        double[:] w   # weight vector
        double[:] q   # sum of squared weights
        double eta    # learning rate (assumed constant)
        double etaC   # = eta*C
        double C      # regularization constant
        int[:] u      # time of last update
        int L         # regularizer type in {1,2}
        int d         # dimensionality
        double fudge  # adagrad fudge factor paramter
        int step      # time step of the optimization algorithm (caller is
                      # responsible for incrementing)

    cdef inline double catchup(self, int k) nogil
    cdef inline void update_active(self, int k, double g) nogil

    cpdef update(self, int[:] keys, double[:] vals)
    cdef inline void _update(self, int[:] keys, double[:] vals) nogil

    cpdef double dot(self, int[:] keys)
    cdef inline double _dot(self, int[:] keys) nogil

    cpdef update_scalar(self, int[:] keys, double v)
    cdef inline void _update_scalar(self, int[:] keys, double v) nogil
