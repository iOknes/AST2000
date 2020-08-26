import numpy as np
import matplotlib.pyplot as plt

from math import pi
from sim import Sim

#Vectorize this, maybe?
def gravity(pos, mass):
    a = np.zeros((len(pos), len(pos),3))
    for i in range(len(pos)):
        for j in range(i+1,len(pos)):
            dr = pos[i] - pos[j]
            r = np.linalg.norm(dr)
            a[i,j] = mass[i] * dr / r**3
            a[j,i] = mass[j] * dr / r**3
    return np.sum(a, axis = 0)

s = Sim()

r = [[0,0,0], [1,0,0]]
v = [[0,0,0], [0,1,0]]
m = [1,0]

s.add_molecules(r, v, m)
s.add_force(gravity)
s.solve(1e-3, 2*pi)

plt.scatter(s.r[0,0,0], s.r[0,0,1], color='b')
plt.plot(s.r[:,0,0], s.r[:,0,1], color='b', label='Particle without initial velocity')
plt.scatter(s.r[0,1,0], s.r[0,1,1], color='r')
plt.plot(s.r[:,1,0], s.r[:,1,1], color='r', label='Particle with initial velocity')
plt.legend()
plt.show()
