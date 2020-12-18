#Egen kode

import numpy as np
from numba import jit

@jit(cache=True, nopython=True)
def normal_gaussian(mu, sigma, x):
    var1 = 1 / (np.sqrt(2*np.pi)*sigma)
    var2 = np.exp((-0.5)*(((x-mu)/sigma)**2))
    return (var1*var2)

@jit
def P(func, a, b, N, mu, sigma):
    x = np.linspace(a, b, N)
    dt = x[1] - x[0]
    p = func(mu, sigma, x) * dt
    return np.sum(p)

@jit
def integrate_probability(func, a, b, N, mu, sigma):
    dx = (b-a)/N
    S = np.zeros(N)
    for i in range(0, N):
        S[i] = dx * (func(mu, sigma, (a+ (dx*i))))
    return (np.sum(S))


if __name__=="__main__":
    sig = 1
    mu = 0

    a = integrate_probability(normal_gaussian,
                              -1*sig, 1*sig, int(1e5), mu, sig)
    b = integrate_probability(normal_gaussian,
                              -2*sig, 2*sig, int(1e5), mu, sig)
    c = integrate_probability(normal_gaussian,
                              -3*sig, 3*sig, int(1e5), mu, sig)

    print(a)
    print(b)
    print(c)

    a1 = P(normal_gaussian, -1*sig, 1*sig, int(1e5), mu, sig)
    b1 = P(normal_gaussian, -2*sig, 2*sig, int(1e5), mu, sig)
    c1 = P(normal_gaussian, -3*sig, 3*sig, int(1e5), mu, sig)
    print("\n")

    print(a1)
    print(b1)
    print(c1)

    """
    python statistics.py
    0.682689492120965
    0.9544997360748356
    0.997300203928758


    0.682694331585032
    0.9545018957367678
    0.9973004698459154
    """
