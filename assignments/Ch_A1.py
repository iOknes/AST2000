import numpy as np

f = lambda m, s, x: np.exp(- ((x - m) / s)**2 / 2) / (np.sqrt(2 * np.pi) * s)

def gauss(stddiv, N):
    x = np.linspace(-stddiv, stddiv, N)
    return np.trapz(f(0, 1, x), x)

#P(a <= x <= b) means the probability of x being between a part of the set [a, b]
print(f"1 stddiv out: {gauss(1, 1001)}")
print(f"2 stddiv out: {gauss(2, 1001)}")
print(f"3 stddiv out: {gauss(3, 1001)}")
