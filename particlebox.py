import numpy as np
from xyz import savexyz

#Particle parameters
pN = 100
pn = round(np.sqrt(pN))
pN = int(pn**2)

#Simulation parameters
dt = 1e-2
N = 1000
l = 1


rx, ry = np.meshgrid(np.arange(pn)/pn, np.arange(pn)/pn)
r_init = np.array([rx.flatten(), ry.flatten()]).transpose()
v_init = np.random.normal(0, 1, (pN, 2))

r = np.zeros((N, pN, 2))
v = np.zeros((N, pN, 2))
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
