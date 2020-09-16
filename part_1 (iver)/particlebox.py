import numpy as np
from modules import xyz, progress
from time import time
from numba import jit
from ast2000tools import utils

#Particle parameters
pN = 10000
sigma = np.sqrt(3000 / 242.466)

#Simulation parameters
dt = 1e-3
N = 1000
l = 1
dim = 3
seed = utils.get_seed("jrevense")
np.random.seed(seed)

t_start = time()

#outfile = xyz.File("particle_box")

r_init = np.random.uniform(0, 1, (pN, dim))
v_init = np.random.normal(0, sigma, (pN, dim))

r = np.zeros((N, pN, dim), dtype="float64")
v = np.zeros((N, pN, dim), dtype="float64")
r[0] = r_init
v[0] = v_init

bar = progress.Bar(N)

def run(r, v, l=1, dt=1e-3, N=1000, nozzle=0.5):
    n_esc = 0
    v_esc = 0
    for i in range(N-1):
        #Euler-Cromer integrator (moving particles)
        v[i+1] = v[i]
        r[i+1] = r[i] + v[i+1] * dt
        #Counting particles that would have escaped
        x_true = (r[i+1, :, 0] < (l / 2 + nozzle / 2)) * (r[i+1, :, 0] > (l / 2 - nozzle / 2))
        y_true = (r[i+1, :, 1] < (l / 2 + nozzle / 2)) * (r[i+1, :, 1] > (l / 2 - nozzle / 2))
        z_true = r[i+1, :, 2] < 0
        n_esc += np.sum(x_true * y_true * z_true)
        v_esc += np.sum(v[i+1, x_true * y_true * z_true, 2])
        #Collison checking and correcting
        v[i+1, r[i+1] < 0] = -v[i+1, r[i+1] < 0]
        v[i+1, r[i+1] > l] = -v[i+1, r[i+1] > l]
        r[i+1, r[i+1] < 0] = -r[i+1, r[i+1] < 0]
        r[i+1, r[i+1] > l] = r[i+1, r[i+1] > l] - 2 * (r[i+1, r[i+1] > l] - l)
        bar()
    return r, v, n_esc, v_esc

r, v, n_esc, v_esc = run(r, v, l, dt, N)

def fuel_consumed(force, fuel_consumption, mass, target_speed):
    pass

print(f"{n_esc} particles escaped and accelerated the box.")
print(f"A total momentum of {v_esc * 3.3476e-24} kg m/s escaped the box.")

print(f"Calculation time: {time() - t_start}s")

#outfile.save(r)
