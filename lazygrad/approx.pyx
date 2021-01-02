#!/usr/bin/env python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: infertypes=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
from libc.stdio cimport printf

cdef inline double sign(double x) nogil:
    return 1 if x >= 0 else -1

cdef inline double abs(double x) nogil:
    return x if x >= 0 else -x


cdef class LazyRegularizedAdagrad:

    def __init__(self, int d, int L, double C, double eta = 0.1, double fudge = 1e-4):
        self.L = L
        self.d = d
        self.fudge = fudge
        self.u = np.zeros(d, dtype=np.int32)
        self.q = np.zeros(d, dtype=np.double) + fudge
        self.w = np.zeros(d, dtype=np.double)
        self.C = C
        self.eta = eta
        self.step = 1
        self.etaC = self.eta*self.C

    def reset(self):
        """ reset the AdaGrad values """
        self.u = np.zeros(self.d, dtype=np.int32)
        self.q = np.zeros(self.d, dtype=np.double) + self.fudge

    def _catchup(self, int k):
        self.catchup(k)

    def _update_active(self, int k, double g):
        self.update_active(k, g)

    def finalize(self):
        for i in range(self.d):
            self.catchup(i)
        return np.asarray(self.w)

    cdef inline double catchup(self, int k) nogil:
        "Lazy L1/L2-regularized adagrad catchup operation."
        cdef int dt
        cdef double s, sq
        dt = self.step - self.u[k]
        # shortcircuit when weights are up-to-date
        if dt == 0:
            return self.w[k]
        s = invsqrt(self.q[k])
        if self.L == 2:
            # Lazy L2 regularization
            self.w[k] *= fastpow(self.etaC * s + 1, -dt)

        elif self.L == 1:
            # Lazy L1 regularization
            self.w[k] = sign(self.w[k]) * max(0, abs(self.w[k]) - self.etaC * dt * s)

        # update last seen
        self.u[k] = self.step
        return self.w[k]

    cdef inline void update_active(self, int k, double g) nogil:
        # Warning: If you call this method multiple times in a time-step you'll
        # get weird results.
        #
        # Possible solutions:
        #
        #  1. Ignore problem - hopefully, gives a sensible asynchronous
        #     approximations.
        #
        #  2. Buffer the gradient updates (or weight vector) for a time step.
        #
        #  3. Other ideas?

        cdef double d, z, s, sq
        self.q[k] += g*g
        s = invsqrt(self.q[k])
        if self.L == 2:
            self.w[k] = (self.w[k] - self.eta * g * s) / (self.etaC * s + 1)

        elif self.L == 1:
            z = self.w[k] - self.eta * g * s
            d = abs(z) - self.etaC * s
            self.w[k] = sign(z) * max(0, d)

        else:
            self.w[k] -= self.eta * g * s

        self.u[k] = self.step+1

    cpdef double dot(self, int[:] keys):
        return self._dot(keys)

    cdef inline double _dot(self, int[:] keys) nogil:
        """
        performs dot product with binary vector (i.e., just the keys)
        Side effect: performs catchup on keys touched.
        """
        cdef int k
        cdef double s = 0.0
        for k in range(keys.shape[0]):
            s += self.catchup(keys[k])
        return s

    cpdef update(self, int[:] keys, double[:] vals):
        assert keys.shape == vals.shape
        self._update(keys, vals)

    cdef inline void _update(self, int[:] keys, double[:] vals) nogil:
        "performs dot product and calls catchup on relevant keys."
        cdef int k, n
        cdef double s = 0.0
        n = keys.shape[0]
        for k in range(n):
            self.update_active(keys[k], vals[k])


def test_L1():
    """
    Integration test for Lazily regularized adagrad.
    """
    import numpy as np
    from numpy import sqrt, sign, zeros

    class EagerL1Weights(object):

        def __init__(self, D, C, a, fudge):
            self.w = zeros(D)
            self.g2 = zeros(D) + fudge
            self.C = C
            self.a = a

        def update(self, g):
            # dense weight update
            self.g2 += g*g
            z = self.w - self.a * g / sqrt(self.g2)
            d = np.abs(z) - self.a*self.C / sqrt(self.g2)
            d[d <= 0] = 0  # d = max(0, d)
            self.w = sign(z) * d

    T = 50  # number of iterations
    D = 6   # number of features
    K = 3   # number of active features

    C = .8        # regularization constant
    eta = .3      # stepsize
    fudge = 1e-4  # adagrad fudge factor

    lazy = LazyRegularizedAdagrad(D, L=1, C=C, eta=eta, fudge=fudge)
    eager = EagerL1Weights(D, C=C, a=eta, fudge=fudge)

    for _ in range(T):

        keys = range(D)
        np.random.shuffle(keys)
        keys = keys[:K]

        # dense vector.
        dense = np.zeros(D)
        dense[keys] = 1
        eager.update(dense)

        for k in keys:
            lazy._catchup(k)
            lazy._update_active(k, 1)

        lazy.step += 1

    print
    print 'step=', lazy.step
    w = np.asarray(lazy.finalize())
    print w
    print eager.w
    err = np.abs(w-eager.w).max()
    assert err < 0.001, err


def test_L2():
    """
    Integration test for Lazily regularized adagrad.
    """
    import numpy as np
    from numpy import sqrt, sign, zeros

    class EagerL2Weights(object):

        def __init__(self, D, C, eta, fudge):
            self.w = zeros(D)
            self.g2 = zeros(D) + fudge
            self.C = C
            self.eta = eta
            self.etaC = eta*C

        def update(self, g):
            # dense weight update
            self.g2 += g*g
            s = 1/np.sqrt(self.g2)
            self.w = (self.w - self.eta * g * s) / (self.etaC * s + 1)


    T = 50  # number of iterations
    D = 6   # number of features
    K = 3   # number of active features

    C = .8        # regularization constant
    eta = .3      # stepsize
    fudge = 1e-4  # adagrad fudge factor

    lazy = LazyRegularizedAdagrad(D, L=2, C=C, eta=eta, fudge=fudge)
    eager = EagerL2Weights(D, C=C, eta=eta, fudge=fudge)

    for _ in range(T):

        keys = range(D)
        np.random.shuffle(keys)
        keys = keys[:K]

        # dense vector.
        dense = np.zeros(D)
        dense[keys] = 1
        eager.update(dense)

        for k in keys:
            lazy._catchup(k)
            lazy._update_active(k, 1)

        lazy.step += 1

    print
    print 'step=', lazy.step
    w = np.asarray(lazy.finalize())

    #from arsenal.math import compare
    #compare(eager.w, w)

    print w
    print eager.w
    err = np.abs(w-eager.w).max()
    assert err < 0.0015, err


from lazygrad cimport adagrad

def test_speed():
    from arsenal.timer import timers
    cdef int i, k

    cdef int I = 50        # number of iterations
    cdef int D = 100_000   # number of features
    cdef int K = 100       # number of active features
    cdef double C = .8     # regularization constant
    cdef long[:] keys
    cdef adagrad.LazyRegularizedAdagrad lazy
    cdef LazyRegularizedAdagrad approx

    T = timers()

    for _ in range(100):

        lazy = adagrad.LazyRegularizedAdagrad(D, L=2, C=C)
        approx = LazyRegularizedAdagrad(D, L=2, C=C)

        for _ in range(I):
            keys = np.random.randint(D, size=K).astype(int)

            with T['lazy']:
                for i in range(K):
                    k = keys[i]
                    lazy.catchup(k)
                    lazy.update_active(k, 1)
                lazy.step += 1

            with T['approx']:
                for i in range(K):
                    k = keys[i]
                    approx.catchup(k)
                    approx.update_active(k, 1)
                approx.step += 1

        err = np.abs(approx.finalize() - lazy.finalize()).max()
        #print 'difference: %g' % err
        assert err < 0.001

    T.compare()


def test():
    test_L1()
    test_L2()
    test_speed()


if __name__ == '__main__':
    test()
