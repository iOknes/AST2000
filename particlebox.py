import numpy as np
from xyz import savexyz

#Particle parameters
pN = 1000

#Simulation parameters
dt = 1e-2
N = 10000
l = 1
dim = 3

r_init = np.random.uniform(0, 1, (pN, dim))
v_init = np.random.normal(0, 1, (pN, dim))

r = np.zeros((N, pN, dim))
v = np.zeros((N, pN, dim))
r[0] = r_init
v[0] = v_init

for i in range(N-1):
    v[i+1] = v[i]
    r[i+1] = r[i] + v[i+1] * dt
    v[i+1, r[i+1] < 0] = -v[i+1, r[i+1] < 0]
    v[i+1, r[i+1] > l] = -v[i+1, r[i+1] > l]
    r[i+1, r[i+1] < 0] = -r[i+1, r[i+1] < 0]
    r[i+1, r[i+1] > l] = r[i+1, r[i+1] > l] - 2 * (r[i+1, r[i+1] > l] - l)

savexyz(r, "particlebox")
